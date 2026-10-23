# 🌊 AquaPINN: Physics-Informed Digital Twin for Wastewater Treatment

AquaPINN is a physics-informed digital twin platform for wastewater engineering. It combines the **Activated Sludge Model No. 1 (ASM1)** kinetic framework with a **PyTorch Physics-Informed Neural Network (PINN)** surrogate model, served through an interactive **Streamlit** dashboard.

---

## 🛠️ Key Features

- **ASM1 Mechanistic Kinetics Engine:** Solves nonlinear differential equations for activated sludge dynamics ($S_S$, $S_O$, $S_{NO}$, $S_{NH}$, $X_{BH}$) with temperature scaling.
- **PyTorch PINN Surrogate:** Fast neural network surrogate constrained by custom mass-balance loss functions to enforce physical consistency.
- **Interactive Digital Twin Dashboard:** Real-time KPI monitoring, scenario testing, Plotly charts, and PINN vs ODE trajectory comparisons.
- **Process Optimization Advisor:** Real-time tracking of effluent quality metrics ($COD$, $NH_4-N$, $DO$, $TN$) under dynamic loading conditions.

---

## 📂 Project Architecture
```
AquaPINN/
├── app.py                   # Main Streamlit Web Application
├── requirements.txt         # Project Dependencies
├── models/                  # Physics & ML Modules
│   ├── __init__.py          # Module Exports
│   ├── asm1_ode.py          # Mechanistic ASM1 Differential Equations
│   └── surrogate_pinn.py    # PyTorch PINN Architecture & Trainer
└── .gitignore               # Ignored local/virtual files
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/goldydagoat/AquaPINN.git](https://github.com/goldydagoat/AquaPINN.git)
cd AquaPINN
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```

---

## 🔬 Physics Constraints & PINN Loss
The neural network minimizes a composite loss function balancing data fidelity and mass balance:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}$$

Where $\mathcal{L}_{\text{physics}}$ penalizes violations of CSTR mass conservation equations across state variables over time.