#!/usr/bin/env python3
"""
Physics Loss Weight (Lambda) Hyperparameter Sweep for AquaPINN Surrogate Model.

This script evaluates the impact of physics loss regularization weights
lambda in [1e-4, 1e-3, 0.01, 0.1] during PINN training (500 epochs).
It computes RMSE and MAE across all 13 ASM1 state variables against
ground-truth ASM1 numerical ODE solutions and serializes benchmark
results to reports/pinn_sweep_results.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.asm1_ode import NUM_STATES, STATE_NAMES, STATE_UNITS, simulate_asm1
from models.surrogate_pinn import ASM1PINN, PINNConfig, PINNTrainer


DEFAULT_LAMBDAS = [1e-4, 1e-3, 0.01, 0.1]
DEFAULT_EPOCHS = 500
DEFAULT_OUTPUT_FILE = "reports/pinn_sweep_results.json"


def prepare_benchmark_dataset(
    days: float = 7.0,
    num_samples: int = 500,
    q_in: float = 1000.0,
    volume: float = 1000.0,
    kla: float = 120.0,
    influent_cod: float = 150.0,
    influent_nh4: float = 35.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ground-truth ASM1 simulation trajectory and surrogate input features.

    Returns:
        (X, Y, t_arr):
            X: Input feature tensor [time, dilution, kla, s_s_in, s_nh_in]
            Y: Target ASM1 states matrix (shape [N, 13])
            t_arr: Time array (days)
    """
    sim_df = simulate_asm1(days=max(days, 7.0), num_points=max(num_samples, 500))
    t_arr = sim_df["time"].values

    d_arr = np.full_like(t_arr, q_in / volume)
    kla_arr = np.full_like(t_arr, kla)
    ss_in_arr = np.full_like(t_arr, influent_cod)
    snh_in_arr = np.full_like(t_arr, influent_nh4)

    X = np.stack([t_arr, d_arr, kla_arr, ss_in_arr, snh_in_arr], axis=-1)
    Y = np.stack([sim_df[name].values for name in STATE_NAMES], axis=-1)

    return X, Y, t_arr


def evaluate_state_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute RMSE and MAE across all 13 ASM1 states and aggregate summary metrics.

    Args:
        y_true: Ground-truth concentration array of shape (N, 13)
        y_pred: Surrogate prediction array of shape (N, 13)

    Returns:
        tuple of (state_metrics_dict, summary_metrics_dict)
    """
    state_metrics: Dict[str, Dict[str, float]] = {}
    rmse_list: List[float] = []
    mae_list: List[float] = []

    for idx, name in enumerate(STATE_NAMES):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        diff = pred_col - true_col

        rmse_val = float(np.sqrt(np.mean(diff ** 2)))
        mae_val = float(np.mean(np.abs(diff)))
        mean_true = float(np.mean(true_col))
        std_true = float(np.std(true_col))
        range_true = float(np.ptp(true_col))

        # Relative RMSE normalized by data range (or mean if range is near zero)
        norm_factor = range_true if range_true > 1e-4 else max(abs(mean_true), 1e-4)
        nrmse_pct = float((rmse_val / norm_factor) * 100.0)

        state_metrics[name] = {
            "rmse": round(rmse_val, 6),
            "mae": round(mae_val, 6),
            "unit": STATE_UNITS.get(name, "g/m³"),
            "nrmse_percent": round(nrmse_pct, 4),
            "mean_ground_truth": round(mean_true, 6),
            "std_ground_truth": round(std_true, 6),
        }

        rmse_list.append(rmse_val)
        mae_list.append(mae_val)

    overall_diff = y_pred - y_true
    summary_metrics = {
        "mean_rmse": round(float(np.mean(rmse_list)), 6),
        "mean_mae": round(float(np.mean(mae_list)), 6),
        "median_rmse": round(float(np.median(rmse_list)), 6),
        "median_mae": round(float(np.median(mae_list)), 6),
        "overall_rmse": round(float(np.sqrt(np.mean(overall_diff ** 2))), 6),
        "overall_mae": round(float(np.mean(np.abs(overall_diff))), 6),
    }

    return state_metrics, summary_metrics


def run_physics_loss_sweep(
    lambdas: List[float] = DEFAULT_LAMBDAS,
    epochs: int = DEFAULT_EPOCHS,
    days: float = 7.0,
    num_samples: int = 500,
    hidden_dim: int = 128,
    device: str = "cpu",
    output_path: Optional[str] = DEFAULT_OUTPUT_FILE,
    save_checkpoints: bool = False,
) -> Dict[str, Any]:
    """
    Execute systematic hyperparameter sweep across physics loss weights (lambda).

    Evaluates training loss convergence, state-wise RMSE/MAE across all 13 states,
    and serializes structured benchmark report.
    """
    print("=" * 80)
    print("🌊 AquaPINN Physics Loss Regularization Weight (Lambda) Sweep")
    print(f"• Evaluated Lambdas: {lambdas}")
    print(f"• Training Epochs : {epochs}")
    print(f"• Simulation Days : {days} (500 collocation / evaluation points)")
    print(f"• Target States   : {NUM_STATES} ASM1 biochemical states")
    print("=" * 80)

    # 1. Prepare ODE ground-truth simulation data
    X, Y, t_arr = prepare_benchmark_dataset(days=days, num_samples=num_samples)

    sweep_results: List[Dict[str, Any]] = []
    by_lambda: Dict[str, Any] = {}

    for i, lam in enumerate(lambdas, start=1):
        lambda_str = f"{lam:g}"
        print(f"\n[{i}/{len(lambdas)}] Training PINN with lambda = {lam:g} (physics_weight={lam}) ...")

        # Configure and instantiate model
        config = PINNConfig(
            input_dim=5,
            hidden_dim=hidden_dim,
            num_layers=4,
            output_dim=NUM_STATES,
            epochs=epochs,
            physics_weight=float(lam),
            learning_rate=3e-3,
            device=device,
        )

        model = ASM1PINN(config)
        trainer = PINNTrainer(model)

        t_start = time.time()
        history = trainer.train(X, Y, epochs=epochs, physics_weight=float(lam))
        training_time = round(time.time() - t_start, 3)

        # Predict concentrations across full simulation horizon
        preds = model.predict(X, t=t_arr)

        # Compute per-state and overall metrics
        state_metrics, summary_metrics = evaluate_state_metrics(Y, preds)

        final_total_loss = float(history["total"][-1]) if history["total"] else 0.0
        final_data_loss = float(history["data"][-1]) if history["data"] else 0.0
        final_physics_loss = float(history["physics"][-1]) if history["physics"] else 0.0

        print(
            f"  └─ Done in {training_time:.2f}s | "
            f"Mean RMSE: {summary_metrics['mean_rmse']:.4f} | "
            f"Mean MAE: {summary_metrics['mean_mae']:.4f} | "
            f"Final Loss: {final_total_loss:.6f} "
            f"(Data: {final_data_loss:.6f}, Physics: {final_physics_loss:.6f})"
        )

        entry = {
            "lambda": float(lam),
            "lambda_str": lambda_str,
            "epochs": epochs,
            "training_time_seconds": training_time,
            "final_losses": {
                "total": round(final_total_loss, 8),
                "data": round(final_data_loss, 8),
                "physics": round(final_physics_loss, 8),
            },
            "summary_metrics": summary_metrics,
            "state_metrics": state_metrics,
        }

        if save_checkpoints:
            ckpt_path = Path("checkpoints") / f"pinn_sweep_lambda_{lambda_str}.pt"
            model.save_checkpoint(ckpt_path)
            entry["checkpoint_path"] = str(ckpt_path)

        sweep_results.append(entry)
        by_lambda[lambda_str] = entry

    # Identify optimal configuration
    best_by_rmse = min(sweep_results, key=lambda res: res["summary_metrics"]["mean_rmse"])
    best_by_mae = min(sweep_results, key=lambda res: res["summary_metrics"]["mean_mae"])

    benchmark_data: Dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/sweep_physics_loss.py",
            "epochs": epochs,
            "lambdas_evaluated": [float(l) for l in lambdas],
            "num_states": NUM_STATES,
            "state_names": STATE_NAMES,
            "state_units": STATE_UNITS,
            "simulation_days": days,
            "num_samples": num_samples,
            "device": device,
        },
        "optimal_configuration": {
            "best_by_mean_rmse": {
                "lambda": best_by_rmse["lambda"],
                "lambda_str": best_by_rmse["lambda_str"],
                "mean_rmse": best_by_rmse["summary_metrics"]["mean_rmse"],
            },
            "best_by_mean_mae": {
                "lambda": best_by_mae["lambda"],
                "lambda_str": best_by_mae["lambda_str"],
                "mean_mae": best_by_mae["summary_metrics"]["mean_mae"],
            },
        },
        "sweep_results": sweep_results,
        "by_lambda": by_lambda,
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Lambda':<10} | {'Mean RMSE':<12} | {'Mean MAE':<12} | {'Data Loss':<14} | {'Phys Loss':<14} | {'Time (s)':<8}"
    print(header)
    print("-" * len(header))
    for res in sweep_results:
        print(
            f"{res['lambda_str']:<10} | "
            f"{res['summary_metrics']['mean_rmse']:<12.4f} | "
            f"{res['summary_metrics']['mean_mae']:<12.4f} | "
            f"{res['final_losses']['data']:<14.6f} | "
            f"{res['final_losses']['physics']:<14.6f} | "
            f"{res['training_time_seconds']:<8.2f}"
        )
    print("-" * len(header))
    print(
        f"🌟 Best Configuration by Mean RMSE: lambda = {best_by_rmse['lambda_str']} "
        f"(RMSE = {best_by_rmse['summary_metrics']['mean_rmse']:.4f})"
    )
    print("=" * 80)

    # Print state-by-state breakdown for best model
    print(f"\n🔬 State-by-State Metrics Breakdown for Best Model (lambda = {best_by_rmse['lambda_str']}):")
    state_hdr = f"{'State':<8} | {'Unit':<15} | {'RMSE':<12} | {'MAE':<12} | {'NRMSE (%)':<10}"
    print(state_hdr)
    print("-" * len(state_hdr))
    for name in STATE_NAMES:
        m = best_by_rmse["state_metrics"][name]
        print(
            f"{name:<8} | "
            f"{m['unit']:<15} | "
            f"{m['rmse']:<12.4f} | "
            f"{m['mae']:<12.4f} | "
            f"{m['nrmse_percent']:<10.2f}%"
        )
    print("=" * 80)

    # Serialize results to reports/
    if output_path:
        out_file = Path(output_path)
        if not out_file.is_absolute():
            if out_file.parts and out_file.parts[0] == "reports":
                out_file = (ROOT_DIR / out_file).resolve()
            else:
                out_file = (Path.cwd() / out_file).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)
        display_path = out_file.relative_to(ROOT_DIR) if out_file.is_relative_to(ROOT_DIR) else out_file
        print(f"\n✅ Benchmark results successfully saved to: {display_path}\n")

    return benchmark_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AquaPINN Physics Loss Weight (Lambda) Hyperparameter Sweep"
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
        help="List of physics loss regularization weights lambda to evaluate (default: 1e-4 1e-3 0.01 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs per lambda configuration (default: 500)",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help="Simulation horizon in days (default: 7.0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of time discretization points (default: 500)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Surrogate network hidden layer dimension (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (cpu or cuda, default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Destination JSON report filepath (default: reports/pinn_sweep_results.json)",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoint for each evaluated lambda configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_physics_loss_sweep(
        lambdas=args.lambdas,
        epochs=args.epochs,
        days=args.days,
        num_samples=args.num_samples,
        hidden_dim=args.hidden_dim,
        device=args.device,
        output_path=args.output,
        save_checkpoints=args.save_checkpoints,
    )


if __name__ == "__main__":
    main()
