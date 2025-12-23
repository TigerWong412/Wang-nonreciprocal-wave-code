README — Source code for “Non-reciprocal Colloidal Transport Guided by Traveling Stimulus Waves”
=============================================================================================

This repository contains the Python source code used to generate simulation results for the
research article:

  “Non-reciprocal Colloidal Transport Guided by Traveling Stimulus Waves”

Questions may be directed to:
  Prof. Wei Wang (Harbin Institute of Technology (Shenzhen))
  Email: weiwangsz@hit.edu.cn


Below is the Readme for 1D simulations. An additional python code, "non_reciprocal_2D.py" is in the same repo that simulates a minimal 2D many-particle model used to calculate the transport speed of SiO2 particles. See Readme_2D.txt for details.


1) Code overview
----------------
Main script:
  - non_reciprocal_1D.py

This script simulates the overdamped dynamics of two interacting particles in one
spatial dimension, subject to a traveling square-wave stimulus. The model is
formulated entirely in nondimensional units and is deterministic by default
(optional thermal noise can be enabled).

Key features:
  - 1D overdamped Brownian dynamics (Euler–Maruyama integration)
  - Traveling square-wave stimulus s(x,t) = sign[sin(2π(x − U t)/λ)]
  - Non-reciprocal pair interaction whose sign depends on the local stimulus
    experienced by the source particle
  - Hard-core contact enforced at one particle diameter
  - Direct visualization of stimulus field and particle motion
  - Automatic generation of videos and diagnostic plots
  - Late-time measurement of the dimensionless transport speed V_T*


2) Units and nondimensionalization
----------------------------------
All quantities in the simulation are nondimensional.

Chosen units:
  - Length unit: particle diameter σ (set to 1)
  - Time unit:   τ (arbitrary simulation time unit)
  - Velocity:    σ / τ

Important note:
  - Forces are already expressed in velocity units.
  - There is no explicit friction coefficient, viscosity, or k_BT in the code.
  - A mapping to physical units can be constructed a posteriori if desired, but is
    not required to reproduce the results shown in the paper.


3) Environment and dependencies
-------------------------------
Recommended Python version:
  - Python >= 3.9

Required Python packages:
  - numpy
  - matplotlib
  - imageio

Optional:
  - ffmpeg executable on PATH (required only for MP4 video generation)

Installation example:
  pip install numpy matplotlib imageio

FFmpeg (macOS with Homebrew):
  brew install ffmpeg


4) How to run
-------------
Run the script from the command line:

  python non_reciprocal_1D.py

The script runs immediately with the default parameters and produces:
  - Two MP4 videos (stimulus + particles, and center-of-mass drift)
  - One PNG summary figure with trajectories and diagnostics
  - Printed estimates of the dimensionless transport speed V_T*


5) Key tunable parameters
-------------------------
The main parameters are defined at the top of the script.

Wave parameters:
  - lam : wavelength λ* (default 10.0)
  - U   : wave speed U* (default 5.0)

Interaction parameters:
  - eps_rep : repulsive amplitude ε_rep*
  - eps_att : attractive amplitude ε_att* (= N * eps_rep)
  - N       : ratio |ε_att| / ε_rep

Noise strength:
  - D : dimensionless translational noise strength
        (D = 0 → deterministic dynamics)

Time integration:
  - dt              : time step
  - n_periods_total : total number of wave periods simulated

Initial conditions:
  - x1, x2 : initial particle positions
    (chosen such that r ≈ 1.25σ and the center-of-mass is at 0)

Video generation:
  - SAVE_VIDEO_FRAMES : enable/disable video output
  - FRAME_RATE_FPS    : output video frame rate
  - dt_frame          : physical time spacing between saved frames


6) What the code generates
--------------------------
If SAVE_VIDEO_FRAMES = True, the following files are produced:

  (A) Video 1:
      active_1D_wave_nondimensional_stimulus.mp4
      Shows the traveling square-wave stimulus and the two particle positions.

  (B) Video 2:
      COM_vs_time.mp4
      Shows the center-of-mass displacement R(t) as a function of time.

Temporary frame folders are created during the run and automatically deleted
after the videos are stitched.

Additional output:
  - bd_pair_nondimensional_summary.png
    A static figure showing particle trajectories, separation, and COM drift.

Console output:
  - Printed estimates of the dimensionless transport speed:
        V_T*
        V_T* / U*


7) Transport speed measurement
------------------------------
The dimensionless transport speed V_T* is computed from the late-time slope of the
center-of-mass trajectory:

  V_T* = ΔR / Δt

The slope is averaged over the last several wave periods to exclude transient
behavior.


8) Reproducibility
------------------
- A fixed random seed is used for the noise generator.
- By default, D = 0 and the dynamics are deterministic.
- To explore stochastic effects, set D > 0.


9) Scope and limitations
------------------------
This 1D pair model is intended as a minimal, illustrative system to reveal the
mechanism of wave-mediated non-reciprocal transport. It neglects:
  - hydrodynamic interactions,
  - many-body effects,
  - higher-dimensional geometry.

For quantitative comparison with experiments and collective effects, see the 2D
many-particle simulation code provided in the same repository.


10) Citation and contact
------------------------
If you use or adapt this code in academic work, please cite:

  “Non-reciprocal Colloidal Transport Guided by Traveling Stimulus Waves”

For questions, contact:
  Prof. Wei Wang
  weiwangsz@hit.edu.cn
