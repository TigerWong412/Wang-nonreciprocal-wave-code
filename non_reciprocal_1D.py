import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import imageio.v2 as iio

# ============================================================
# 1. Dimensionless units and parameters
# ------------------------------------------------------------
# Length unit: particle diameter σ
# Time unit:  τ (arbitrary; velocities are in σ/τ)
# All variables below are nondimensional.
# ============================================================

sigma = 1.0           # particle diameter (unit length)
a = 0.5 * sigma       # particle radius

# Wave parameters (dimensionless)
lam = 10.0            # wavelength λ* = λ / σ
U = 5               # wave speed U* = U τ / σ

# Interaction parameters
N = 1.0               # |eps_att| / eps_rep
eps_rep =1         # repulsive amplitude ε_rep*
eps_att = N * eps_rep # attractive amplitude ε_att*

# Dimensionless translational noise strength D*
# (set to 0 for deterministic dynamics)
D = 0.0

# Simulation time: number of wave periods and timestep
n_periods_total = 10
T_period = lam / U              # one wave period in t*
t_total = n_periods_total * T_period
dt = 1e-3                       # timestep in t*
nsteps = int(np.ceil(t_total / dt))

# ------------------------------------------------------------
# Video generation parameters
# ------------------------------------------------------------
SAVE_VIDEO_FRAMES = True

# Video 1: waves and particles
OUTPUT_VIDEO_NAME_WAVE = "active_1D_wave_nondimensional_stimulus.mp4"
FRAMES_DIR_WAVE = "video_frames_1D_nondimensional_stimulus"

# Video 2：COM vs time
OUTPUT_VIDEO_NAME_COM = "COM_vs_time.mp4"
FRAMES_DIR_COM = "video_frames_COM"

FRAME_RATE_FPS = 30

# a fixed dt is used so waves appear fast or slow in the generated videos
dt_frame = 0.2
FRAME_SKIP = max(1, int(round(dt_frame / dt)))

# Initial positions: separation ≈ 1.25 σ, center-of-mass at 0
x2 = 0.625 * sigma
x1 = -0.625 * sigma

rng = np.random.default_rng(1234)

# ============================================================
# 2. Helper functions
# ============================================================

def square_wave_sign(phase: float) -> float:
    """
    Dimensionless square wave: +1 on 'bright' half, -1 on 'dark' half.
    """
    s = np.sin(phase)
    return 1.0 if s >= 0.0 else -1.0

def stimulus_sign(x: float, t: float) -> float:
    """
    Stimulus field s(x,t) = sign[sin(2π(x - U t)/λ)] in dimensionless units.
    """
    phase = 2.0 * np.pi * (x - U * t) / lam
    return square_wave_sign(phase)

def enforce_contact(x1: float, x2: float) -> tuple[float, float]:
    """
    Enforce hard-core contact: r >= 1 (two radii).
    Implemented by fixing the center-of-mass and projecting to r = 1 if needed.
    """
    r = x2 - x1
    if r < sigma:
        R = 0.5 * (x1 + x2)
        x1 = R - 0.5 * sigma
        x2 = R + 0.5 * sigma
    return x1, x2

# ============================================================
# 3. Allocate arrays and set initial state
# ============================================================

ts = np.empty(nsteps + 1)
R_hist = np.empty(nsteps + 1)
r_hist = np.empty(nsteps + 1)
x1_hist = np.empty(nsteps + 1)
x2_hist = np.empty(nsteps + 1)

t = 0.0
ts[0] = t
x1, x2 = enforce_contact(x1, x2)
R_hist[0] = 0.5 * (x1 + x2)
r_hist[0] = x2 - x1
x1_hist[0] = x1
x2_hist[0] = x2

# ============================================================
# 4. Video setup
# ============================================================

if SAVE_VIDEO_FRAMES:
    # generate Video 1
    if os.path.exists(FRAMES_DIR_WAVE):
        shutil.rmtree(FRAMES_DIR_WAVE)
    os.makedirs(FRAMES_DIR_WAVE)

    # Generate video 2
    if os.path.exists(FRAMES_DIR_COM):
        shutil.rmtree(FRAMES_DIR_COM)
    os.makedirs(FRAMES_DIR_COM)

    # ---------- Video 1 ----------
    fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
    ax_wave.set_xlim(-5*sigma, 5*sigma)
    ax_wave.set_ylim(-1.2, 1.2)
    ax_wave.set_xlabel('Position x / σ')
    ax_wave.set_ylabel('stimulus')  
    ax_wave.set_title('Stimulus field and particles')

    x_wave = np.linspace(-5*sigma, 5*sigma, 1000)
    wave_line, = ax_wave.plot([], [], 'b-', alpha=0.5, linewidth=2, label='stimulus')

    
    particle1, = ax_wave.plot([x1], [0.0], 'ko', markersize=8, label='Particle 1')
    particle2, = ax_wave.plot([x2], [0.0], 'ro', markersize=8, label='Particle 2')

    ax_wave.legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 1.0),
    frameon=True
)
    ax_wave.grid(True, alpha=0.3)

    # ---------- Video 2：COM vs time ----------
    fig_com, ax_com = plt.subplots(figsize=(8, 4))
    ax_com.set_xlim(0.0, t_total)
    ax_com.set_ylim(-5*sigma, 5*sigma)  
    ax_com.set_xlabel('Time / τ')
    ax_com.set_ylabel('Center-of-mass R / σ')
    ax_com.set_title('Center-of-mass drift')

    com_line, = ax_com.plot([], [], 'k-', linewidth=2)

    ax_com.grid(True, alpha=0.3)

# ============================================================
# 5. Time integration (overdamped BD in dimensionless form)
# ============================================================

frame_count_wave = 0
frame_count_com = 0

for k in range(1, nsteps + 1):
    # Local stimulus states
    s1 = stimulus_sign(x1, t)
    s2 = stimulus_sign(x2, t)

    # Separation and hard-core constraint
    r = x2 - x1
    if r < sigma:
        x1, x2 = enforce_contact(x1, x2)
        r = x2 - x1

    # Non-reciprocal pair forces (already in velocity units)
    F1_from_2 = (-eps_rep if s2 > 0 else eps_att) / r**2
    F2_from_1 = ( eps_rep if s1 > 0 else -eps_att) / r**2

    v1 = F1_from_2
    v2 = F2_from_1

    # Dimensionless thermal noise (optional)
    if D > 0.0:
        noise1 = np.sqrt(2.0 * D * dt) * rng.standard_normal()
        noise2 = np.sqrt(2.0 * D * dt) * rng.standard_normal()
    else:
        noise1 = 0.0
        noise2 = 0.0

    # Euler–Maruyama update in nondimensional variables
    x1 += v1 * dt + noise1
    x2 += v2 * dt + noise2

    # Enforce hard-core again after the update
    x1, x2 = enforce_contact(x1, x2)

    # Record history
    t += dt
    ts[k] = t
    x1_hist[k] = x1
    x2_hist[k] = x2
    R_hist[k] = 0.5 * (x1 + x2)
    r_hist[k] = x2 - x1
    
    # Save video frames
    if SAVE_VIDEO_FRAMES and (k % FRAME_SKIP == 0 or k == nsteps):

        # ---------- updating Video 1 ----------
        wave_vals = np.array([stimulus_sign(x, t) for x in x_wave])   
        wave_line.set_data(x_wave, wave_vals)

        particle1.set_data([x1], [0.0])
        particle2.set_data([x2], [0.0])

        ax_wave.set_title(f'Stimulus and particles (t = {t:.2f} τ)')

        frame_path_wave = os.path.join(
            FRAMES_DIR_WAVE, f"frame_{frame_count_wave:05d}.png"
        )
        fig_wave.savefig(frame_path_wave, dpi=100, bbox_inches='tight')
        frame_count_wave += 1

        # ---------- updating Video 2：COM vs time ----------
        com_line.set_data(ts[:k+1], R_hist[:k+1])
        ax_com.set_title(f'Center-of-mass drift (t = {t:.2f} τ)')

        frame_path_com = os.path.join(
            FRAMES_DIR_COM, f"frame_{frame_count_com:05d}.png"
        )
        fig_com.savefig(frame_path_com, dpi=100, bbox_inches='tight')
        frame_count_com += 1

        if frame_count_wave % 50 == 0:
            print(f"  Saved {frame_count_wave} wave frames, {frame_count_com} COM frames...")

# ============================================================
# 6. Estimate dimensionless transport speed V_T*
# ============================================================

Np = 5                          # average over last Np periods
T = T_period
t_end = ts[-1]
t_start = t_end - Np * T
mask = ts >= t_start

if np.sum(mask) == 0 or (ts[mask][-1] - ts[mask][0]) == 0:
    VT = 0.0
else:
    VT = (R_hist[mask][-1] - R_hist[mask][0]) / (ts[mask][-1] - ts[mask][0])

print(f"Dimensionless V_T* ≈ {VT:.4f}")
print(f"V_T*/U* ≈ {VT / U:.4f}")

# ============================================================
# 7. Stitch videos from frames
# ============================================================

if SAVE_VIDEO_FRAMES:
    plt.close(fig_wave)
    plt.close(fig_com)

    # ---------- stiching video 1：stimulus + particles ----------
    print(f"Stitching wave video '{OUTPUT_VIDEO_NAME_WAVE}' with {frame_count_wave} frames...")
    frame_files_wave = sorted(
        os.path.join(FRAMES_DIR_WAVE, f) for f in os.listdir(FRAMES_DIR_WAVE)
        if f.endswith('.png')
    )
    frames_wave = [iio.imread(f) for f in frame_files_wave]
    iio.mimsave(OUTPUT_VIDEO_NAME_WAVE, frames_wave,
                fps=FRAME_RATE_FPS, quality=8)

    # ---------- stiching video 2：COM vs time ----------
    print(f"Stitching COM video '{OUTPUT_VIDEO_NAME_COM}' with {frame_count_com} frames...")
    frame_files_com = sorted(
        os.path.join(FRAMES_DIR_COM, f) for f in os.listdir(FRAMES_DIR_COM)
        if f.endswith('.png')
    )
    frames_com = [iio.imread(f) for f in frame_files_com]
    iio.mimsave(OUTPUT_VIDEO_NAME_COM, frames_com,
                fps=FRAME_RATE_FPS, quality=8)

    # clearing temp folders
    shutil.rmtree(FRAMES_DIR_WAVE)
    shutil.rmtree(FRAMES_DIR_COM)

    print(f"Videos saved as '{OUTPUT_VIDEO_NAME_WAVE}' and '{OUTPUT_VIDEO_NAME_COM}'. Temporary frames removed.")

# ============================================================
# 8. Simple diagnostic plots (all in dimensionless units)
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

axes[0].plot(ts, x1_hist, label='x1')
axes[0].plot(ts, x2_hist, label='x2')
axes[0].set_ylabel('Position / σ')
axes[0].set_title('Particle trajectories')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(ts, r_hist)
axes[1].axhline(sigma, color='k', linestyle=':', linewidth=1)
axes[1].set_ylabel('Separation r / σ')
axes[1].set_title('Separation (contact at r = 1)')
axes[1].grid(alpha=0.3)

axes[2].plot(ts, R_hist)
axes[2].set_xlabel('Time / τ')
axes[2].set_ylabel('Center-of-mass R / σ')
axes[2].set_title(f'Center-of-mass drift; V_T*/U* ≈ {VT/U:.3f}')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bd_pair_nondimensional_summary.png', dpi=200)
# plt.show()
