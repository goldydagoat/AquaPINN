# AGENT.md: Google Antigravity Subagent Guidelines for AquaPINN

## 🌊 Project Overview
AquaPINN is a Level 3 predictive digital twin for wastewater treatment plants. It combines:
1. **ASM1 First-Principles ODE Kinetics** (`models/asm1_ode.py`): Nonlinear differential equations for activated sludge dynamics ($S_S, S_O, S_{NO}, S_{NH}, X_{BH}$, etc.) with temperature scaling.
2. **PyTorch Physics-Informed Neural Networks (PINNs)** (`models/surrogate_pinn.py`): Neural network surrogate with per-variable MinMaxScaler, explicit time input normalization ($t \in [0, 1]$, time multiplier $5.0$), SiLU activations for smooth second-order derivatives $\frac{d^2\hat{y}}{dt^2}$, and dynamic physics-data loss balancing.
3. **Interactive Streamlit Dashboard & Plotly** (`app.py`, `data_handler.py`): Real-time control inputs, SCADA/BSM1 data ingestion, and interactive time-series plots.

---

## 🛠️ Subagent Operating Principles

### 1. Development Lifecycle
- **Research:** Systematically map codebase dependencies and inspect relevant modules (`app.py`, `models/asm1_ode.py`, `models/surrogate_pinn.py`, `data_handler.py`).
- **Strategy:** Formulate a minimal, robust, and idiomatic implementation plan adhering to workspace patterns.
- **Execution:** Apply surgical changes using available tools (`replace`, `write_file`, `run_shell_command`).
- **Validation:** Always verify syntax compilation (`python -m py_compile app.py models/surrogate_pinn.py models/asm1_ode.py data_handler.py`) and empirical correctness (running training / simulation tests).

### 2. Code & Architecture Standards
- **Physics Core (`models/asm1_ode.py`):** Maintain rigorous stoichiometric and kinetic balance parameters for ASM1 ODE equations.
- **Surrogate Core (`models/surrogate_pinn.py`):** Preserve per-variable feature scaling, explicit time input scaling ($t \times 5.0$), SiLU activation functions, dynamic loss balancing, and strict deterministic seeding (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `torch.cuda.manual_seed_all(42)`).
- **Dashboard (`app.py`):** Use Streamlit for control inputs and Plotly for interactive time-series plots. Ensure session state (`active_pinn_model`, `pinn_trained`) is correctly maintained without unnecessary checkpoint re-loads.
- **Formatting & Style:** Keep documentation clean (Markdown). Use LaTeX math strings only for formal mathematical formulations (e.g., $\frac{dy}{dt}$). Adhere strictly to typing, type safety, and existing workspace conventions without bypassing type checkers or linters.

### 3. Verification Rules
- **Syntax Check:** Always run `python -m py_compile app.py models/surrogate_pinn.py models/asm1_ode.py data_handler.py` before finalizing changes.
- **Testing & Validation:** Execute smoke tests or training verification scripts (`.venv/bin/python -c "from models.surrogate_pinn import train_pinn; train_pinn(days=1.0, num_samples=50, epochs=5)"`).
- **Git Hygiene:** Do not stage or commit changes unless explicitly instructed by the user. When commit instructions are given, review `git status` and `git diff` first, stage specific files, and provide a clear commit message.
