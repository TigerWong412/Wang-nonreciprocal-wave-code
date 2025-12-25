import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd
from numba import njit
import shutil
import time
import sys
from datetime import datetime


# Prevent errors on headless servers (no display available)
plt.switch_backend('Agg')

# ==========================================
# 1. Numba-accelerated kernel
#    (Skin-distance formulation retained; force-vector normalization corrected)
# ==========================================

@njit(cache=True)
def compute_forces_kernel(
    r, diameter, is_active, is_tracer, in_wave, wave_type,
    ag_diameter, epsilon_wca, cutoff_sat_NR,
    eps_dark, eps_bright,
    cut_nr, neighbor_pairs,
    Lx, Ly
):
    N = r.shape[0]
    F = np.zeros((N, 2), dtype=np.float64)

    ag2 = ag_diameter * ag_diameter
    cut2_nr = cut_nr * cut_nr

    # Set a minimum cutoff distance (saturation distance) to prevent division by zero
    # and excessively large forces
    SATURATION_DISTANCE = cutoff_sat_NR * ag_diameter

    for idx in range(neighbor_pairs.shape[0]):
        i = neighbor_pairs[idx, 0]
        j = neighbor_pairs[idx, 1]

        # --- A. Basic geometry (minimum-image convention under PBC) ---
        dx = r[i, 0] - r[j, 0]
        dy = r[i, 1] - r[j, 1]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)

        r2_center = dx * dx + dy * dy
        if r2_center <= 1e-12:
            continue

        r_ij = np.sqrt(r2_center)
        inv_r = 1.0 / r_ij

        # [Fix] Normalized direction vector n
        nx = dx * inv_r
        ny = dy * inv_r

        # --- B. WCA repulsive force ---
        sigma = 0.5 * (diameter[i] + diameter[j])
        r_cut_wca = (2.0 ** (1.0 / 6.0)) * sigma

        if r_ij < r_cut_wca:
            sig_r = sigma * inv_r
            sig2_r2 = sig_r * sig_r
            sig6_r6 = sig2_r2 * sig2_r2 * sig2_r2
            f_mag_wca = 48.0 * epsilon_wca * inv_r * sig6_r6 * (sig6_r6 - 0.5)
            if f_mag_wca > 1000.0:
                f_mag_wca = 1000.0

            fx = f_mag_wca * nx
            fy = f_mag_wca * ny
            F[i, 0] += fx
            F[i, 1] += fy
            F[j, 0] -= fx
            F[j, 1] -= fy

        # --- C. Active non-reciprocal force (skin-distance based) ---
        if r2_center > cut2_nr:
            continue

        skin_distance = r_ij - sigma
        if skin_distance < SATURATION_DISTANCE:
            constrained_dist = SATURATION_DISTANCE
        else:
            constrained_dist = skin_distance

        dist_sq = constrained_dist * constrained_dist
        factor = ag2 / dist_sq

        # Case 1: i is the source
        if is_active[i] and in_wave[i]:
            mag = eps_dark if wave_type[i] == 0 else -eps_bright
            F[j, 0] += mag * factor * nx
            F[j, 1] += mag * factor * ny

        # Case 2: j is the source
        if is_active[j] and in_wave[j]:
            mag = eps_dark if wave_type[j] == 0 else -eps_bright
            F[i, 0] -= mag * factor * nx
            F[i, 1] -= mag * factor * ny

    return F

@njit(cache=True)
def build_neighbor_list_numba(r, Lx, Ly, cutoff):
    N = r.shape[0]
    cutoff2 = cutoff * cutoff
    max_pairs = N * 500
    pairs = np.empty((max_pairs, 2), dtype=np.int64)
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = r[i, 0] - r[j, 0]
            dy = r[i, 1] - r[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            if dx * dx + dy * dy <= cutoff2:
                if count < max_pairs:
                    pairs[count, 0] = i
                    pairs[count, 1] = j
                    count += 1
                else:
                    return pairs[:count], True
    return pairs[:count], False

# ==========================================
# 2. Simulation class
# ==========================================

class Simulation:
    def __init__(self,
                 dt=0.001,
                 max_time=10.0,
                 after_when=1.0,
                 ag_diameter=1.0,
                 sio2_diameter=5.0,
                 n_active=400,
                 n_tracer=5,
                 box_Lx=100.0,
                 box_Ly=100.0,
                 eta=1.0,
                 kBT=1.0,
                 epsilon_wca=10.0,
                 noise_on=True,
                 wave_speed=3.0,
                 wavelength=30.0,
                 duty_cycle=0.5,
                 cut_nr=10.0,
                 eps_dark=0.1,
                 cutoff_sat_NR=0.5,
                 beta=1.0,
                 print_every_time=0.1,
                 verlet_skin=1.0,
                 seed=None,
                 output_dir="temp_data",
                 save_snapshots=True):

        self.dt = dt
        self.max_time = max_time
        self.after_when = after_when
        self.ag_diameter = ag_diameter
        self.sio2_diameter = sio2_diameter
        self.n_active = int(n_active)
        self.n_tracer = int(n_tracer)
        self.N = self.n_active + self.n_tracer
        self.Lx = box_Lx
        self.Ly = box_Ly
        self.eta = eta
        self.kBT = kBT
        self.epsilon_wca = epsilon_wca
        self.noise_on = noise_on
        self.cutoff_sat_NR = cutoff_sat_NR

        self.wave_speed = wave_speed
        self.wavelength = wavelength
        self.dark_width = (1.0 - duty_cycle) * self.wavelength
        self.cut_nr = cut_nr
        self.verlet_skin = verlet_skin

        self.eps_dark = eps_dark
        self.eps_bright = beta * eps_dark

        self.print_every_steps = max(1, int(print_every_time / dt))
        self.rng = np.random.default_rng(seed)

        self.output_dir = output_dir
        self.save_snapshots = save_snapshots
        if self.save_snapshots:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        self.r = np.zeros((self.N, 2), dtype=np.float64)
        self.r_last_build = np.zeros((self.N, 2), dtype=np.float64)
        self.dr_cum = np.zeros((self.N, 2), dtype=np.float64)
        self.diameter = np.zeros(self.N, dtype=np.float64)
        self.D = np.zeros(self.N, dtype=np.float64)
        self.gamma = np.zeros(self.N, dtype=np.float64)
        self.is_active = np.zeros(self.N, dtype=np.bool_)
        self.is_tracer = np.zeros(self.N, dtype=np.bool_)
        self.neighbor_pairs = np.empty((0, 2), dtype=np.int64)

        self.history_t = []
        self.history_tracer_x = []

        self._init_particles()
        self._init_diffusion()

        max_sigma = max(ag_diameter, sio2_diameter)
        self.cutoff_global = max((2.0**(1.0/6.0)) * max_sigma, self.cut_nr)
        self.cutoff_verlet = self.cutoff_global + self.verlet_skin
        self._rebuild_verlet()
        actual_area = self.Lx * self.Ly
        particle_area = self.n_active * np.pi * (self.ag_diameter / 2) ** 2
        print(f"Actual area fraction: {particle_area / actual_area:.4f}")

    def _init_particles(self):
        """
        Improved initialization function:
        1. Use a large-particle-first strategy (sorted indices).
        2. Increase the attempt limit to 100,000.
        3. Use PBC (periodic boundary condition) distance checks to ensure
           no overlap occurs near the boundaries.
        """
        # Assign particle roles: first n_active are Ag, the rest are SiO2
        self.diameter[:self.n_active] = self.ag_diameter
        self.diameter[self.n_active:] = self.sio2_diameter
        self.is_active[:self.n_active] = True
        self.is_tracer[self.n_active:] = True

        # Place particles in descending order of diameter (SiO2 first, then Ag)
        sorted_indices = np.argsort(self.diameter)[::-1]

        # Mask of particles already placed
        placed_mask = np.zeros(self.N, dtype=np.bool_)

        # Leave a small margin near boundaries to avoid force-evaluation issues (optional)
        margin = 0.01 * self.ag_diameter

        for i in sorted_indices:
            placed = False
            curr_diam = self.diameter[i]

            # Increase attempts to 100,000 to handle ~33% area fraction
            for attempt in range(100000):
                # Random trial position
                x = self.rng.random() * self.Lx
                y = self.rng.random() * self.Ly

                # If this is the first particle, accept directly
                if not np.any(placed_mask):
                    self.r[i] = (x, y)
                    placed = True
                    break

                # Indices of particles already placed
                existing_indices = np.where(placed_mask)[0]

                # Minimum-image distances to all placed particles (PBC)
                dx = self.r[existing_indices, 0] - x
                dy = self.r[existing_indices, 1] - y

                # Key step: apply minimum-image convention (Periodic Boundary Conditions)
                dx -= self.Lx * np.round(dx / self.Lx)
                dy -= self.Ly * np.round(dy / self.Ly)

                dist2 = dx * dx + dy * dy

                # Allowed minimum center-to-center distance squared
                min_dist = 0.5 * (self.diameter[existing_indices] + curr_diam)
                min_dist2 = min_dist * min_dist

                # Successful placement if far enough from all existing particles
                if np.all(dist2 >= min_dist2):
                    self.r[i] = (x, y)
                    placed = True
                    break

            if not placed:
                # If still not placed after 100,000 attempts, warn and force placement
                print(f"Warning: Particle {i} (D={curr_diam}) could not find a slot after 100,000 attempts. Forced overlap.")
                self.r[i] = (self.rng.random() * self.Lx, self.rng.random() * self.Ly)

            placed_mask[i] = True

    def _init_diffusion(self):
        radius = 0.5 * self.diameter
        self.D = self.kBT / (6.0 * np.pi * self.eta * radius)
        self.gamma = self.kBT / self.D

    def _rebuild_verlet(self):
        pairs, overflow = build_neighbor_list_numba(self.r, self.Lx, self.Ly, self.cutoff_verlet)
        if overflow:
            print("Warning: Neighbor list overflow!")
        self.neighbor_pairs = pairs
        self.r_last_build = self.r.copy()

    def _check_verlet_update(self):
        dr = self.r - self.r_last_build
        dr[:, 0] -= self.Lx * np.round(dr[:, 0] / self.Lx)
        dr[:, 1] -= self.Ly * np.round(dr[:, 1] / self.Ly)
        max_disp = np.max(np.sqrt(np.sum(dr**2, axis=1)))
        if max_disp > 0.5 * self.verlet_skin:
            self._rebuild_verlet()

    def _dump_snapshot(self, time):
        fname = os.path.join(self.output_dir, f"{time:.6f}.txt")
        colors = np.zeros(self.N)
        colors[self.is_tracer] = 3
        data = np.column_stack((
            self.r[:, 0], self.r[:, 1],
            0.5 * self.diameter, colors,
            self.dr_cum[:, 0], self.dr_cum[:, 1]
        ))
        np.savetxt(fname, data, fmt="%.6f")

    def run(self):
        steps = int(self.max_time / self.dt)
        time_sim = 0.0
        step = 0
        noise_scale = np.sqrt(2.0 * self.D * self.dt)

        import sys  # Ensure imported inside the method or at file top
        start_wall_time = time.time()

        while step <= steps:
            self._check_verlet_update()

            wave_pos = (self.wave_speed * time_sim) % self.Lx
            dx_wave = (self.r[:, 0] - wave_pos + self.Lx) % self.Lx
            x_mod = np.mod(dx_wave, self.wavelength)
            in_dark = x_mod < self.dark_width
            wave_type = np.where(in_dark, 0, 1)

            F = compute_forces_kernel(
                self.r, self.diameter, self.is_active, self.is_tracer,
                np.ones(self.N, dtype=np.bool_),
                wave_type,
                self.ag_diameter, self.epsilon_wca, self.cutoff_sat_NR,
                self.eps_dark, self.eps_bright,
                self.cut_nr, self.neighbor_pairs,
                self.Lx, self.Ly
            )

            drift = (self.dt / self.gamma)[:, None] * F

            if self.noise_on:
                xi = self.rng.standard_normal((self.N, 2))
                dx = drift + noise_scale[:, None] * xi
            else:
                dx = drift

            self.r += dx
            self.dr_cum += dx
            self.r[:, 0] = np.mod(self.r[:, 0], self.Lx)
            self.r[:, 1] = np.mod(self.r[:, 1], self.Ly)

            if step % self.print_every_steps == 0:
                self.history_t.append(time_sim)
                self.history_tracer_x.append(self.dr_cum[self.is_tracer, 0].copy())

                # --- Corrected progress display logic ---
                elapsed_wall_time = time.time() - start_wall_time
                progress = step / steps

                if progress > 0:
                    total_estimated_time = elapsed_wall_time / progress
                    remaining_time = total_estimated_time - elapsed_wall_time

                    msg = (f"\r    [Progress: {progress*100:5.1f}%] "
                           f"Steps: {step}/{steps} | "
                           f"Elapsed: {elapsed_wall_time:5.1f}s | "
                           f"Remaining: {remaining_time:5.1f}s    ")
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                else:
                    # If progress is too small to estimate remaining time, print steps only
                    sys.stdout.write(f"\r    [Progress: initializing...] Steps: {step}/{steps}")
                    sys.stdout.flush()

                if self.save_snapshots:
                    self._dump_snapshot(time_sim)

            step += 1
            time_sim += self.dt

        # End-of-simulation handling
        sys.stdout.write("\r" + " " * 100 + "\r")  # Clear the progress bar line with spaces
        sys.stdout.flush()
        print(f"Simulation finished! Total wall time: {time.time() - start_wall_time:.2f}s")
        return np.array(self.history_t), np.array(self.history_tracer_x)


# ==========================================
# 3. Utility: video generation
# ==========================================
def generate_video_from_frames(sim_params, data_dir, output_filename):
    Lx = sim_params["box_Lx"]
    Ly = sim_params["box_Ly"]
    wavelength = sim_params["wavelength"]
    wave_speed = sim_params["wave_speed"]
    duty_cycle = sim_params.get("duty_cycle", 0.5)
    dark_width = (1.0 - duty_cycle) * wavelength

    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")),
                   key=lambda f: float(os.path.splitext(os.path.basename(f))[0]))
    if not files:
        return

    frames = []
    # --- 1. Update: dynamically compute the canvas aspect ratio ---
    # Set a reference width (e.g., 15 inches), and scale the height by the physical aspect ratio
    base_width = 15.0
    fig_w = base_width
    fig_h = base_width * (Ly / Lx)

    # Ensure image dimensions are multiples of 16 pixels (for ffmpeg compatibility)
    dpi = 100
    # Slightly adjust height so that (fig_h * dpi) becomes an even / multiple-of-16 number
    fig_h = np.round(fig_h * dpi / 16) * 16 / dpi

    for i, fname in enumerate(files):
        if i % 2 != 0:
            continue
        try:
            t = float(os.path.splitext(os.path.basename(fname))[0])
            data = np.loadtxt(fname)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        except:
            continue

        xs, ys, rads, cols = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.set_aspect('equal')  # lock physical aspect ratio

        # --- Updated wave-background drawing logic ---
        wave_pos = (wave_speed * t) % Lx
        # Compute the minimum number of periods needed to cover the whole canvas.
        # Start k from -2 to ensure background exists when the left boundary wraps;
        # extend the ending range so the right boundary is always covered.
        num_periods = int(np.ceil(Lx / wavelength)) + 2
        for k in range(-2, num_periods):
            base = wave_pos + k * wavelength

            # Only draw if part of the dark region overlaps [0, Lx], to improve rendering efficiency
            dark_start = base
            dark_end = base + dark_width

            # Use axvspan's internal clipping, but restrict to visible range
            if dark_end > 0 and dark_start < Lx:
                ax.axvspan(max(0, dark_start), min(Lx, dark_end),
                           color='0.85', zorder=0, linewidth=0)
        # ---------------------------------------

        # Pixels-per-coordinate-unit conversion
        # 72 is Matplotlib's default DPI conversion constant
        points_per_unit = (72.0 * fig_w) / Lx

        # Correct s formula: s = (physical radius * 2 * conversion_factor)^2
        # Note: Matplotlib's "s" is not strictly diameter^2;
        # empirically, the following gives good visual matching (0.6 is subjective to look realistic).
        ax.scatter(xs, ys, s=(rads * 2 * points_per_unit)**2 * 0.6,
                   linewidths=0, c=cols, cmap='viridis', edgecolors='k')

        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.axis('off')

        # Tight layout to reduce whitespace
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        # Get RGBA buffer and convert to RGB array
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(output_filename, frames, fps=10, macro_block_size=16)


# ==========================================
# 4. Main controller: parameter sweep
# ==========================================

def run_parameter_sweep(scan_config, noise_enabled=True, generate_video=False, save_data=True, unit_params=None):
    # --- Dimensionless baseline parameters (center points) ---
    from datetime import datetime
    # Baseline example: wavelength=30, wave_speed=10, box_size=60, n_active=1000
    base_params = {
        "n_active": 1000, "n_tracer": 15, "beta": 1.0,
        "box_size": 60, "eps_dark": 5,
        "wavelength": 30, "wave_speed": 30, "sio2_diameter": 5.0
    }

    # Fixed configuration
    static_config = {
        "dt": 0.001,  #  the simulation with 0.001 is very close to 0.0005.
        "max_time": 1,
        "after_when": 0.1,
        "ag_diameter": 1.0,
        "eta": 0.25,
        "kBT": 1.0,
        "noise_on": noise_enabled,
        "print_every_time": 0.1,
        "epsilon_wca": 10.0,
        "cutoff_sat_NR": 0.5
    }

    # --- Automatically build sweep tasks ---
    tasks = []

    # Helper: generate a log-spaced range around the center value
    def make_range(center_val, num=10):
        # Avoid the (unlikely) case center is 0
        if center_val == 0:
            return np.linspace(0, 5, num)
        low = center_val / 6
        high = center_val * 1
        return np.geomspace(low, high, num)

    if not any(scan_config.values()):
        print("Running base configuration (No Sweep).")
        tasks.append(("BASE_RUN", 0, base_params.copy()))
    else:
        # 1. Number of Ag particles (integer)
        if scan_config.get("n_active", False):
            base_v = base_params["n_active"]
            # Particle count must be integer; unique to avoid duplicates
            vals = np.unique(make_range(base_v).astype(int))
            for v in vals:
                p = base_params.copy()
                p["n_active"] = v
                tasks.append(("n_active", v, p))

        # 2. Number of SiO2 particles (integer)
        if scan_config.get("n_tracer", False):
            base_v = base_params["n_tracer"]
            vals = np.unique(make_range(base_v).astype(int))
            # Ensure at least 1
            vals = vals[vals > 0]
            for v in vals:
                p = base_params.copy()
                p["n_tracer"] = v
                tasks.append(("n_tracer", v, p))

        # 3. Beta
        if scan_config.get("beta", False):
            base_v = base_params["beta"]
            for v in make_range(base_v):
                p = base_params.copy()
                p["beta"] = v
                tasks.append(("beta", v, p))

        # 4. Box size
        if scan_config.get("box_size", False):
            base_v = base_params["box_size"]
            for v in make_range(base_v):
                p = base_params.copy()
                p["box_size"] = v
                tasks.append(("box_size", v, p))

        # 5. eps_dark
        if scan_config.get("eps_dark", False):
            base_v = base_params["eps_dark"]
            for v in make_range(base_v):
                p = base_params.copy()
                p["eps_dark"] = v
                tasks.append(("eps_dark", v, p))

        # 6. Wavelength
        if scan_config.get("wavelength", False):
            base_v = base_params["wavelength"]
            for v in make_range(base_v):
                p = base_params.copy()
                p["wavelength"] = v
                tasks.append(("wavelength", v, p))

        # 7. Wave speed
        if scan_config.get("wave_speed", False):
            base_v = base_params["wave_speed"]
            for v in make_range(base_v):  # use this if you want log-spaced values
            # for v in np.linspace(1.0, 3.0, 3):  # use this if you want a few linear values
                p = base_params.copy()
                p["wave_speed"] = v
                tasks.append(("wave_speed", v, p))

        # 8. SiO2 diameter
        if scan_config.get("sio2_diameter", False):
            # Update: no longer use make_range (log sweep); use linear 1.0 to 10.0
            for v in np.linspace(1.0, 10.0, 10):
                p = base_params.copy()
                p["sio2_diameter"] = v
                tasks.append(("sio2_diameter", v, p))

    # --- Execute sweep ---
    plot_dir = "sweep_plots"
    os.makedirs(plot_dir, exist_ok=True)
    results_list = []

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    scale_factor_velocity = 1.0
    unit_suffix = " (Sim Units)"

    if unit_params:
        R_real_um = unit_params['R_real_um']
        T_real_K = unit_params['T_real_K']
        eta_real_PaS = unit_params['eta_real_PaS']

        kB_SI = 1.380649e-23
        T = T_real_K
        eta = eta_real_PaS
        R = R_real_um * 1e-6

        # Real diffusion coefficient: D = kT / (6 pi eta R)
        D_real_SI = (kB_SI * T) / (6 * np.pi * eta * R)
        D_real_um2s = D_real_SI * 1e12

        # Diffusion coefficient in the simulation
        # Note: here R_sim = 0.5 * ag_diameter = 0.5
        D_sim = static_config['kBT'] / (6 * np.pi * static_config['eta'] * 0.5 * static_config['ag_diameter'])

        # Length scaling: 1.0 sim unit = 2 * R_real_um (Ag diameter)
        L_real_um = R_real_um * 2.0

        # Velocity scaling: v_real = v_sim * (D_real / D_sim) / L_real
        scale_factor_velocity = (D_real_um2s / D_sim) / L_real_um
        unit_suffix = " (um/s)"

        print(f"\n[Unit conversion enabled]")
        print(f"  Real D: {D_real_um2s:.4g} um^2/s")
        print(f"  Velocity scaling factor: x{scale_factor_velocity:.4f}")

    for idx, (param_name, val, run_config) in enumerate(tasks):
        # 1. Remove conflicting parameters and recompute Lx, Ly
        run_config.pop("box_size", None)
        current_wavelength = run_config.get("wavelength", base_params["wavelength"])
        Lx_val = current_wavelength * 2.0
        Ly_val = 90.0

        # --- Corrected logic: only recompute n_active by area when not scanning n_active ---
        if param_name != "n_active":
            phi = 0.2182
            new_n_active = int((phi * Lx_val * Ly_val) / (np.pi * 0.25))
            run_config["n_active"] = new_n_active
        else:
            # If scanning n_active, use the task value
            new_n_active = val
            run_config["n_active"] = int(val)

        # Temporary folder path
        temp_vid_dir = f"temp_{param_name}_{val:.2f}"

        print(f"\n>>> Task [{idx+1}/{total_tasks}]: {param_name}={val:.4f} | Lx={Lx_val:.1f}, N_Ag={new_n_active}")

        try:
            # 2. Initialize and run simulation
            sim = Simulation(
                box_Lx=Lx_val,
                box_Ly=Ly_val,
                output_dir=temp_vid_dir,
                save_snapshots=generate_video,  # <--- linked behavior: only save snapshots if video is enabled
                seed=42,
                **run_config,
                **static_config
            )

            t_hist, x_hist = sim.run()

            # 3. Result analysis (mean velocity of individual SiO2 tracers and its standard deviation)
            mask = t_hist > static_config["after_when"]
            if np.sum(mask) < 2:
                mask = np.arange(len(t_hist)) > len(t_hist) - 5
            valid_t = t_hist[mask]
            valid_x = x_hist[mask, :]

            dt_valid = valid_t[-1] - valid_t[0]
            v_per_particle = (valid_x[-1, :] - valid_x[0, :]) / (dt_valid if dt_valid > 1e-6 else 1.0)
            avg_v_sim = np.mean(v_per_particle)
            std_v_sim = np.std(v_per_particle)

            avg_v_real = avg_v_sim * scale_factor_velocity
            std_v_real = std_v_sim * scale_factor_velocity

            # 4. Video generation logic
            if generate_video:
                print(f"    Stitching video...")
                vid_name = os.path.join(plot_dir, f"{param_name}_{val:.2f}.mp4")
                # Prepare plotting parameters
                v_params = run_config.copy()
                v_params.update(static_config)
                v_params["box_Lx"] = Lx_val
                v_params["box_Ly"] = Ly_val

                # Call the function you just debugged manually
                generate_video_from_frames(v_params, temp_vid_dir, vid_name)

                # Delete temp folder after successful stitching
                if os.path.exists(vid_name):
                    shutil.rmtree(temp_vid_dir, ignore_errors=True)
                print(f"    Video saved to: {vid_name}")

            # 5. Store results dictionary (fixes earlier L_val error)
            res_entry = {
                "Ag_N": new_n_active,
                "SiO2_N": run_config.get("n_tracer", base_params["n_tracer"]),
                "beta": run_config.get("beta", base_params["beta"]),
                "Box_Size": f"{Lx_val:.1f}x{Ly_val:.1f}",  # or store only Lx_val
                "eps_dark": run_config.get("eps_dark", base_params["eps_dark"]),
                "wavelength": current_wavelength,
                "wave_speed": run_config.get("wave_speed", base_params["wave_speed"]),
                "SiO2_Diam": run_config.get("sio2_diameter", base_params["sio2_diameter"]),
                "epsilon_wca": static_config["epsilon_wca"],
                "cutoff_sat_NR": static_config["cutoff_sat_NR"],
                "Avg_Velocity": avg_v_real,   # converted physical-unit velocity
                "Std_Dev": std_v_real,        # keep std for plotting
                "param_name": param_name,     # helper: which parameter is swept
                "param_val": val              # helper: swept parameter value
            }
            results_list.append(res_entry)
            print(f"--- Task finished: V = {avg_v_real:.3f} ± {std_v_real:.3f} {unit_suffix}")

        except Exception as e:
            print(f"    [ERROR] Task {param_name}={val} failed: {e}")
            # Clean up leftover folder on error
            if generate_video and os.path.exists(temp_vid_dir):
                shutil.rmtree(temp_vid_dir)

    # === Export and plotting after the sweep ===
    if save_data and results_list:
        df = pd.DataFrame(results_list)

        # 1. Identify which sweep parameters were activated
        active_params = [k for k, v in scan_config.items() if v]
        if not active_params:
            param_tag = "BASE_RUN"
            xlabel_name = "Task Index"
        else:
            param_tag = "_".join(active_params)
            xlabel_name = active_params[0]  # use the swept parameter as x-axis label
            # 2. Sort (now we have "param_val", so it won't error)
            df = df.sort_values(by="param_val")

        # 3. Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"results_{param_tag}_{timestamp}"
        csv_filename = f"{base_filename}.csv"
        img_filename = f"{base_filename}.png"

        # 4. Save CSV
        df.to_csv(csv_filename, index=False)
        print(f"\n[Data export] Results saved to: {csv_filename}")

        # 5. Summary trend plot
        try:
            plt.figure(figsize=(10, 6), dpi=100)

            # --- Key fix: explicitly use the column names defined in res_entry ---
            y_col = "Avg_Velocity"
            y_err_col = "Std_Dev" if "Std_Dev" in df.columns else None

            if y_err_col:
                plt.errorbar(df["param_val"], df[y_col], yerr=df[y_err_col],
                             fmt='-o', color='#1f77b4', ecolor='gray',
                             capsize=5, elinewidth=1.5, markeredgewidth=1.5,
                             label=f'Mean ± Std (N={run_config["n_tracer"]})')
            else:
                plt.plot(df["param_val"], df[y_col], '-o', color='#1f77b4')

            plt.xlabel(f"Swept Parameter: {xlabel_name}", fontsize=12)
            plt.ylabel(f"Transport Velocity {unit_suffix}", fontsize=12)
            plt.title(f"Transport Velocity vs {xlabel_name}", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best')

            plt.tight_layout()
            plt.savefig(img_filename)
            print(f"[Trend plot] Summary figure saved to: {img_filename}")

        except Exception as plot_error:
            print(f"[Warning] Failed to generate summary trend plot: {plot_error}")

    else:
        print("\n[Note] CSV not saved: results list is empty.")

if __name__ == "__main__":
    # ====================================================
    # 1. Sweep switches (True = sweep enabled, False = use fixed value)
    # ====================================================
    SCAN_SWITCHES = {
        "n_active":      False,
        "n_tracer":      False,
        "beta":          False,
        "box_size":      False,
        "eps_dark":      False,
        "wavelength":    False,
        "wave_speed":    False,
        "sio2_diameter": False
    }

    # ====================================================
    # 2. Feature toggles
    # ====================================================
    ENABLE_NOISE   = True    # Enable Brownian noise?
    GENERATE_VIDEO = False    # Generate videos?
    SAVE_DATA      = True    # Save CSV?

    # ====================================================
    # 3. Real-world parameters (for converting dimensionless results to SI units)
    # ====================================================
    UNIT_PARAMS = {
        'R_real_um': 0.5,      # Real radius of Ag particle (um), i.e., diameter is 1.0 um
        'T_real_K': 298.0,     # Temperature (K)
        'eta_real_PaS': 0.001  # Water viscosity (Pa*s)
    }

    run_parameter_sweep(
        SCAN_SWITCHES,
        noise_enabled=ENABLE_NOISE,
        generate_video=GENERATE_VIDEO,
        save_data=SAVE_DATA,
        unit_params=UNIT_PARAMS
    )
