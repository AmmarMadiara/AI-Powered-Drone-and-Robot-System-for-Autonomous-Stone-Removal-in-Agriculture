# Optimization Model

## Overview

This folder contains an energy-aware multi-trip vehicle routing optimisation model for stone collection. The model determines how stones can be collected and transported while considering vehicle payload capacity, travel time, and battery energy consumption.

The objective is to minimise the total collection time while satisfying the vehicle capacity and energy constraints. Different model formulations are also tested to compare their computational performance.

## Scripts

- **`main.py`** – Main script that coordinates the optimisation process, runs the experiments, and saves the results.
- **`config.py`** – Defines the input data, depot location, vehicle parameters, energy settings, solver settings, and experiment configurations.
- **`data.py`** – Processes the input data, calculates distances between stones and the depot, and prepares the optimisation instance.
- **`model.py`** – Builds the mathematical optimisation model, including routing, payload capacity, travel time, and energy constraints.
- **`reporting.py`** – Extracts the optimised routes and generates detailed reports for each solution.
- **`experiment_tracking.py`** – Records solver progress and performance metrics during each optimisation run.
- **`performance_plots.py`** – Generates plots to visualise solver performance and convergence.
- **`impact_analysis.py`** – Compares the different model formulations against the baseline model.

## Input Data

The model uses an Excel file named `data.xlsx`, containing the stone identifiers, coordinates (`x`, `y`), and mass (`mass`). The input file and column names are specified in `config.py`.

## Workflow

The workflow starts by loading the input data and configuration, followed by preparing the optimisation instance and building the mathematical model. Gurobi then solves the model, and the resulting routes and performance metrics are saved. Finally, the experiment results are used to generate plots and compare the different model formulations.

## Outputs

The model generates optimisation solution files (`.sol`), individual solution reports (`.txt`), experiment and convergence records (`results/`), and performance and comparison plots (`plots/`).

## Running the Model

Run the main script from this folder:

```bash
python main.py
```

A working Gurobi installation and valid licence are required.

**Note:** Rename `experiment_tracking(1).py`, `performance_plots(1).py`, and `impact_analysis(1).py` to remove `(1)`, since the scripts import these modules using the filenames without the suffix.