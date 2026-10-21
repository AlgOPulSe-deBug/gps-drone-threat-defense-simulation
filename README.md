# GPS Drone Threat Defense Simulation

**Type:** Academic, simulation-only project  
**Language:** Python  
**Libraries:** NumPy, Matplotlib, Pillow

## Overview

This project studies a simplified two-dimensional drone-navigation scenario using synthetic GPS measurement noise, an inertial/dead-reckoning estimate, residual-threshold monitoring, and visual analysis. It compares several simulated cases and produces plots and an animation for academic review.

**Important scope limitation:** This is a mathematical simulation, not a real-world counter-drone system. It does not transmit or interfere with GPS signals, control physical aircraft, or establish real-world defensive effectiveness. All coordinates and sensor measurements are synthetic.

## Features

- Discrete-time 2D point-mass navigation model
- Synthetic GPS measurement noise and IMU/dead-reckoning drift
- Residual-threshold anomaly monitoring
- Comparison of simulated scenarios
- Spatial harm-map visualization
- Static plots and an animated GIF
- Command-line options for selecting simulation coordinates
- Fixed random seeds for repeatable experiments within a compatible software environment

## Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- Pillow (used for GIF export)

## Quick start — Windows Command Prompt

Open Command Prompt and move to the folder where you cloned or extracted the repository:

```bat
cd C:\path\to\gps-drone-threat-defense-simulation
```

Create a virtual environment:

```bat
py -m venv .venv
```

Activate it:

```bat
.venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run with interactive prompts:

```bat
python simulation.py
```

Run with default coordinates without prompts:

```bat
python simulation.py --non-interactive
```

Run with custom coordinates:

```bat
python simulation.py --start 100 100 --city 800 800
```

If the `py` launcher is unavailable, try `python` in place of `py` when creating the virtual environment. If `python` is not recognized, install Python from the official Python website and enable the option to add Python to PATH.

## macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python simulation.py
```

## Outputs

When the script completes, it saves generated figures and the animation in `outputs/`. The directory is created automatically. Generated files are excluded from Git by default; run the simulation to recreate them locally.

## Repository layout

```text
gps-drone-threat-defense-simulation/
├── simulation.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── docs/
│   └── project-report.pdf
└── outputs/
    └── .gitkeep
```

## Reproducibility and limitations

- The model is a simplified 2D mathematical abstraction, not a validated aircraft model.
- Sensor noise, drift, thresholds, and map values are simulated assumptions.
- A fixed random seed improves repeatability, but results may vary across software versions and environments.
- Do not interpret synthetic results as proof of real-world system performance.
- No real GPS transmitter, receiver, aircraft, or external navigation system is controlled by this code.

## Academic context

The accompanying report is included in `docs/project-report.pdf`. Please verify that the report's numerical results match the current code before citing them, because code and report versions can differ.

## License

Released under the MIT License. See `LICENSE`.
