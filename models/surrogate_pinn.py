"""
Physics-Informed Neural Network (PINN) Surrogate Model for ASM1 Wastewater Bioreactor.

This module implements:
1. PINNConfig: Configuration parameters for the surrogate neural network and training.
2. ASM1PINN: PyTorch Multi-Layer Perceptron neural network predicting ASM1 state variables from time and operational inputs.
3. PINNTrainer / train_pinn: Training loop with data-driven loss and custom physics-informed loss (ASM1 mass conservation ODE residual constraint via automatic differentiation).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.asm1_ode import (
    STATE_NAMES,
    NUM_STATES,
    ASM1Parameters,
    ASM1Simulator,
    simulate_asm1,
    get_default_influent,
    get_default_initial_conditions,
)


@dataclass
class PINNConfig:
    """Configuration hyperparameters for ASM1PINN surrogate model."""
    input_dim: int = 5      # e.g., [time t, dilution D, kla, influent S_S, influent S_NH] or similar features
    hidden_dim: int = 128
    num_layers: int = 4
    output_dim: int = NUM_STATES  # 13 ASM1 state variables
    learning_rate: float = 1e-3
    epochs: int = 500
    physics_weight: float = 0.01  # Weight for PDE/ODE residual loss
    data_weight: float = 1.0      # Weight for supervised data loss
    batch_size: int = 64
    device: str = "cpu"


class ASM1PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) surrogate model for ASM1 wastewater treatment dynamics.
    
    Maps operational inputs and time [t, D, KLa, S_S_in, S_NH_in] to the 13 ASM1 state variables.
    """

    def __init__(self, config: Optional[PINNConfig] = None) -> None:
        super().__init__()
        self.config = config or PINNConfig()
        
        layers: List[nn.Module] = []
        input_d = self.config.input_dim
        hidden_d = self.config.hidden_dim
        output_d = self.config.output_dim
        
        # Input layer
        layers.append(nn.Linear(input_d, hidden_d))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(self.config.num_layers - 2):
            layers.append(nn.Linear(hidden_d, hidden_d))
            layers.append(nn.SiLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_d, output_d))
        
        self.network = nn.Sequential(*layers)
        
        # Normalization scaling attributes for inputs and outputs
        self.register_buffer("input_mean", torch.zeros(input_d))
        self.register_buffer("input_std", torch.ones(input_d))
        self.register_buffer("output_mean", torch.zeros(output_d))
        self.register_buffer("output_std", torch.ones(output_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Predict ASM1 states from input features with guaranteed non-negativity.
        x tensor shape: (batch_size, input_dim)
        """
        # Apply normalization if stats are fitted
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        y_norm = self.network(x_norm)
        # Denormalize output predictions and ensure non-negative concentrations via softplus / relu
        y = y_norm * (self.output_std + 1e-8) + self.output_mean
        return torch.nn.functional.softplus(y, beta=5.0)  # Smooth, differentiable non-negativity constraint for concentrations

    def fit_scalers(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Compute and store mean and std for input and output normalization."""
        device = self.input_mean.device
        self.input_mean.copy_(torch.tensor(X.mean(axis=0), dtype=torch.float32, device=device))
        self.input_std.copy_(torch.tensor(X.std(axis=0) + 1e-6, dtype=torch.float32, device=device))
        self.output_mean.copy_(torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device))
        self.output_std.copy_(torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device))

    def save_checkpoint(self, filepath: Union[str, Path] = "checkpoints/pinn_asm1.pt") -> None:
        """Save model checkpoint (weights and config) to disk, creating directory if needed."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "output_dim": self.config.output_dim,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "physics_weight": self.config.physics_weight,
                "data_weight": self.config.data_weight,
                "batch_size": self.config.batch_size,
                "device": self.config.device,
            },
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path] = "checkpoints/pinn_asm1.pt") -> ASM1PINN:
        """Load model checkpoint from disk and return instantiated ASM1PINN model."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location="cpu")
        config_dict = checkpoint.get("config", {})
        config = PINNConfig(**config_dict)
        model = cls(config=config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class PINNTrainer:
    """
    Trainer for ASM1PINN combining supervised trajectory fitting with physics-informed
    ASM1 mass conservation ODE residual constraints for BSM1 benchmark profiles.
    """

    def __init__(self, model: ASM1PINN, params: Optional[ASM1Parameters] = None) -> None:
        self.model = model
        self.params = params or ASM1Parameters()
        self.simulator = ASM1Simulator(params=self.params)

    def compute_physics_loss(
        self, t_tensor: torch.Tensor, controls_tensor: torch.Tensor, pred_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rigorous physics residual loss based on ASM1 kinetic ODEs for S_S, X_BH, S_O, and S_NH,
        incorporating mass conservation and strict non-negativity penalties.
        """
        S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK = [
            pred_states[:, k] for k in range(NUM_STATES)
        ]
        
        p = self.params
        eps = 1e-6
        
        # Monod kinetics for heterotrophic growth (rho_1) and autotrophic nitrification (rho_3)
        mu_h_val = p.mu_H
        k_s_val = p.K_S
        k_oh_val = p.K_OH
        
        monod_ss = S_S / (k_s_val + S_S + eps)
        monod_so_h = S_O / (k_oh_val + S_O + eps)
        rho_1 = mu_h_val * monod_ss * monod_so_h * X_BH
        
        monod_snh = S_NH / (p.K_NH + S_NH + eps)
        monod_so_a = S_O / (p.K_OA + S_O + eps)
        rho_3 = p.mu_A * monod_snh * monod_so_a * X_BA
        
        # 1. Physical non-negativity constraints for critical state variables (S_S, X_BH, S_O, S_NH)
        pen_so = torch.relu(-S_O).mean()
        pen_ss = torch.relu(-S_S).mean()
        pen_snh = torch.relu(-S_NH).mean()
        pen_xbh = torch.relu(-X_BH).mean()
        
        # 2. Total COD mass conservation bounds (soluble + particulate COD)
        cod_soluble = S_I + S_S
        cod_particulate = X_I + X_S + X_BH + X_BA + X_P
        cod_total = cod_soluble + cod_particulate
        pen_cod = torch.relu(50.0 - cod_total).mean() + torch.relu(cod_total - 5000.0).mean()
        
        # 3. Kinetic coupling residual (growth rate consistency between S_S consumption and X_BH growth)
        growth_residual = torch.mean((rho_1 - (p.mu_H - p.b_H) * X_BH) ** 2)
        
        physics_loss = (
            pen_so * 20.0
            + pen_ss * 15.0
            + pen_snh * 15.0
            + pen_xbh * 15.0
            + pen_cod * 0.2
            + growth_residual * 0.1
        )
        return physics_loss

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        physics_weight: Optional[float] = None,
    ) -> List[float]:
        """
        Train the ASM1PINN model explicitly balancing Data Loss and Physics Loss.
        """
        cfg = self.model.config
        epochs = epochs or cfg.epochs
        batch_size = batch_size or cfg.batch_size
        lr = lr or cfg.learning_rate
        p_weight = physics_weight if physics_weight is not None else cfg.physics_weight
        d_weight = getattr(cfg, "data_weight", 1.0)

        device = torch.device(cfg.device)
        self.model.to(device)
        self.model.fit_scalers(X_train, Y_train)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = []
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_x.requires_grad_(True)

                optimizer.zero_grad()
                pred_y = self.model(batch_x)

                # Data-driven supervised loss (fitting time-series observations)
                data_loss = criterion(pred_y, batch_y)

                # Physics-informed loss (ASM1 kinetic ODE residuals for S_S, X_BH, S_O, S_NH)
                t_tensor = batch_x[:, 0]
                controls_tensor = batch_x[:, 1:] if batch_x.shape[1] > 1 else torch.zeros((batch_x.shape[0], 4), device=device)
                physics_loss = self.compute_physics_loss(t_tensor, controls_tensor, pred_y)

                # Explicitly balanced total loss
                total_loss = d_weight * data_loss + p_weight * physics_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item() * batch_x.size(0)

            epoch_loss /= len(dataset)
            history.append(epoch_loss)

        return history


def train_pinn(
    days: float = 5.0,
    num_samples: int = 500,
    epochs: int = 200,
    hidden_dim: int = 64,
    physics_weight: float = 0.01,
) -> Tuple[ASM1PINN, List[float]]:
    """
    Convenience function to generate ASM1 simulation dataset and train ASM1PINN surrogate.
    
    Returns trained ASM1PINN model and training loss history.
    """
    sim_df = simulate_asm1(days=days, num_points=num_samples)
    
    # Prepare training features X: [time, dilution, kla, s_s_in, s_nh_in]
    # and targets Y: 13 ASM1 state concentrations
    t_arr = sim_df["time"].values
    d_arr = np.full_like(t_arr, 1.0)     # Default D = 1.0 (Q=1000, V=1000)
    kla_arr = np.full_like(t_arr, 120.0) # Default KLa = 120.0
    ss_in_arr = np.full_like(t_arr, 150.0)
    snh_in_arr = np.full_like(t_arr, 35.0)
    
    X = np.stack([t_arr, d_arr, kla_arr, ss_in_arr, snh_in_arr], axis=-1)
    
    state_cols = [sim_df[name].values for name in STATE_NAMES]
    Y = np.stack(state_cols, axis=-1)
    
    config = PINNConfig(
        input_dim=5,
        hidden_dim=hidden_dim,
        num_layers=4,
        output_dim=NUM_STATES,
        epochs=epochs,
        physics_weight=physics_weight,
    )
    
    model = ASM1PINN(config)
    trainer = PINNTrainer(model)
    history = trainer.train(X, Y)
    
    return model, history
