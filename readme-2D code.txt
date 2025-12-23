README — Source code for “Non-reciprocal Colloidal Transport Guided by Traveling Stimulus Waves”
=============================================================================================

This repository contains the Python source code used to generate simulation results for the
research article:

  “Non-reciprocal Colloidal Transport Guided by Traveling Stimulus Waves”

Questions may be directed to:
  Prof. Wei Wang (Harbin Institute of Technology (Shenzhen))
  Email: weiwangsz@hit.edu.cn


Below is the Readme for 2D simulations. An additional python code, "non_reciprocal_1D.py" is in the same repo that simulates a minimal 1D two-particle model used to illustrate the transport mechanism. See Readme_1D.txt for details.


1) Code overview
----------------
Main script (2D many-particle BD + parameter sweep):
  - non_reciprocal_2D.py

This script runs overdamped Brownian-dynamics simulations in 2D with periodic boundaries for:
  - Active Ag particles (sources of non-reciprocal interactions)
  - Passive SiO2 tracer particles (cargo)

Key model components:
  - WCA steric repulsion
  - Short-ranged non-reciprocal interaction based on skin distance (surface separation),
    with a saturation distance to avoid force divergence
  - Traveling square-wave stimulus along x, controlling the sign/magnitude of NR interactions
  - Verlet-style neighbor list (rebuilt adaptively)
  - Optional Brownian noise
  - Automatic parameter sweep over selected variables
  - Optional video generation from saved frames
  - CSV export and summary plot for transport velocity vs swept parameter
  - Optional conversion of simulation velocities to physical units (µm/s) by diffusion matching


2) Environment and dependencies
-------------------------------
Recommended Python version:
  - Python >= 3.9 (3.10/3.11 recommended)

Required Python packages:
  - numpy
  - numba
  - matplotlib
  - pandas
  - imageio

Optional (only needed if GENERATE_VIDEO=True):
  - ffmpeg executable available on PATH (used by imageio)

Installation example (pip):
  pip install numpy numba matplotlib pandas imageio

FFmpeg (macOS with Homebrew):
  brew install ffmpeg


3) How to run
-------------
Run the 2D parameter-sweep script from the command line:

  python non_reciprocal_2D.py

The script is configured from the __main__ block (bottom of the file). The most important
controls are:

  (A) Which parameter to sweep:
      SCAN_SWITCHES = {
        "n_active":      False,
        "n_tracer":      False,
        "beta":          False,
        "box_size":      False,
        "eps_dark":      False,
        "wavelength":    False,
        "wave_speed":    False,
        "sio2_diameter": True
      }

      Set exactly one (or more) entries to True to sweep those parameters.
      If all are False, the code runs a single “base configuration”.

  (B) Feature toggles:
      ENABLE_NOISE   = True    # Brownian noise on/off
      GENERATE_VIDEO = True    # Generate MP4 videos (also enables snapshot saving)
      SAVE_DATA      = True    # Save CSV + summary plot

  (C) Physical-unit conversion (optional):
      UNIT_PARAMS = {
        'R_real_um': 0.5,      # Real Ag radius (µm) → Ag diameter = 1.0 µm
        'T_real_K': 298.0,     # Temperature (K)
        'eta_real_PaS': 0.001  # Viscosity (Pa·s)
      }

      If UNIT_PARAMS is provided, the script converts velocities from simulation units to µm/s
      by matching diffusion coefficients (Stokes–Einstein scaling).


4) Key tunable parameters (most relevant)
-----------------------------------------
In run_parameter_sweep(), the code defines a baseline parameter set and a static configuration:

Baseline sweep “center points” (base_params):
  - n_active       : number of Ag particles (note: may be recomputed to keep area fraction fixed)
  - n_tracer       : number of SiO2 tracer particles
  - sio2_diameter  : SiO2 diameter (in Ag-diameter units)
  - wavelength     : wavelength of traveling stimulus (in simulation length units)
  - wave_speed     : speed of traveling stimulus (in simulation length/time units)
  - eps_dark       : NR interaction strength in dark region
  - beta           : eps_bright = beta * eps_dark
  - box_size       : used as a sweep label; actual Lx may be tied to wavelength in the sweep logic

Static configuration (static_config):
  - dt             : integration time step
  - max_time       : total simulation time
  - after_when     : transient time excluded from velocity measurement
  - ag_diameter    : Ag diameter (default 1.0 → sets length unit)
  - eta, kBT       : set diffusion in simulation units
  - epsilon_wca    : WCA strength
  - cut_nr         : NR interaction cutoff (passed through run_config)
  - cutoff_sat_NR  : saturation distance factor (prevents NR divergence)
  - verlet_skin    : neighbor-list skin (controls rebuild frequency)

Notes on the sweep logic:
  - In the sweep loop, Lx is set to 2*wavelength and Ly is fixed (Ly=90).
  - n_active is recomputed to maintain a target area fraction (phi).
  - These choices are part of the study design; edit them if you want a different protocol.


5) What the code generates (outputs)
------------------------------------
Depending on the toggles, the script generates:

(A) CSV summary (if SAVE_DATA=True):
  - results_<PARAM>_<timestamp>.csv
    Columns include:
      param_name, param_val, Ag_N, Box_Lx, Box_Ly,
      Avg_Velocity_Sim, Std_Velocity_Sim,
      Avg_Velocity (um/s) and Std_Velocity (um/s) if UNIT_PARAMS enabled

(B) Summary plot (if SAVE_DATA=True):
  - results_<PARAM>_<timestamp>.png
    Errorbar plot of mean ± std transport velocity vs swept parameter

(C) Videos (if GENERATE_VIDEO=True):
  - sweep_plots/<param_name>_<param_val>.mp4
    The code stores temporary snapshot frames in per-run folders named:
      temp_<param_name>_<param_val>
    After successful MP4 creation, the temporary folder is deleted automatically.

(D) Console logs:
  - Progress messages for each sweep task, including computed velocities and unit conversion.


6) Expected runtime and performance notes
-----------------------------------------
- The force computation is accelerated with Numba (@njit). The first run may be slower due to
  compilation; subsequent runs are faster.
- Video generation can dominate runtime and disk I/O. For physics-only sweeps, set:
    GENERATE_VIDEO = False
  This disables snapshot saving and MP4 generation.


7) Reproducibility and random seeds
-----------------------------------
- The Simulation object accepts a seed; the sweep uses seed=42 by default for reproducibility.
- If you want independent replicates, vary the seed and/or run multiple repeats per parameter.


8) Citation
-----------
If you use or adapt this code in academic work, please cite the associated article:
  “Non-reciprocal Colloidal Transport Guided by Traveling Stimulus Waves”

For questions, contact:
  Prof. Wei Wang — weiwangsz@hit.edu.cn
