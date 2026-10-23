"""AquaPINN Model Package: Physics-Informed Digital Twin for Wastewater Treatment."""

from models.asm1_ode import ASM1Parameters, ASM1Simulator, simulate_asm1
from models.surrogate_pinn import ASM1PINN, PINNTrainer, PINNConfig

__all__ = [
    "ASM1Parameters",
    "ASM1Simulator",
    "simulate_asm1",
    "ASM1PINN",
    "PINNTrainer",
    "PINNConfig",
]
