# AquaPINN Developer Guidelines

## Project Context
AquaPINN is a Level 3 predictive digital twin for wastewater treatment plants. It combines ASM1 first-principles ODE kinetics with PyTorch Physics-Informed Neural Networks (PINNs) and an interactive Streamlit dashboard.

## Project Structure Rules
- Always maintain a single, flat root structure (`app.py`, `data_handler.py`, `models/`, `requirements.txt`).
- The primary virtual environment is located at `.venv/` in the repository root.
- Never create or reference nested subfolders like `AquaPINN/AquaPINN`.
- Exclude `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`, and weight artifacts (`*.pt`, `*.pth`) from Git tracking.

## Code & Architecture Standards
- **Physics Core (`models/asm1_ode.py`):** Enforce stoichiometric and kinetic balance parameters for ASM1 ODE equations.
- **Surrogate Core (`models/surrogate_pinn.py`):** Use PyTorch autograd for automatic differentiation and residual calculation in loss functions.
- **Dashboard (`app.py`):** Use Streamlit for control inputs and Plotly for interactive time-series plots.
- **Formatting:** Keep standard text clean (Markdown). Use LaTeX math strings only for formal mathematical formulations (e.g., $\frac{dy}{dt}$).

## Operational Instructions
- Validate file paths relative to the root directory before running commands.
- Run tests or check syntax before proposing commits.