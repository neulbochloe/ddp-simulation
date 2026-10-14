# Dependent Dirichlet Process Simulations

Simulation code for studying frequentist coverage of Bayesian credible intervals under dependent Dirichlet process models.

## Requirements

- Python 3
- C++17 compiler
- Python packages listed in requirements.txt

Install dependencies:

    python -m pip install -r requirements.txt

## Compile the C++ module (macOS)

    c++ -O3 -Wall -shared -std=c++17 -fPIC \
      -undefined dynamic_lookup \
      $(python -m pybind11 --includes) \
      ddp_cpp_fast.cpp \
      -o ddp_cpp_fast$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

## Run simulations

    python ddp_final_parallel.py

## Generate figures

    python ddp_draw_linear_figures.py --help

## Files

- ddp_final_parallel.py: Main simulation script
- ddp_stage2c_optimized_sempgy.py: Simulation utilities
- ddp_cpp_fast.cpp: C++ Gibbs sampler
- ddp_draw_linear_figures.py: Figure generation
