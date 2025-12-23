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

# Prevent errors on headless servers (no display)
plt.switch_backend('Agg')

# ==========================================
# 1. Numba-accelerated force kernel
#    (Skin-distance formulation, corrected force normalization)
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

    # Minimum saturation distance to avoid divergence and division by zero
    SATURATION_DISTANCE = cutoff_sat_NR * ag_diameter

    for idx in range(neighbor_pairs.shape[0]):
        i = neighbor_pairs[idx, 0]
        j = neighbor_pairs[idx, 1]

        # --- A. Geometry with minimum-image convention (PBC) ---
        dx = r[i, 0] - r[j, 0]
        dy = r[i, 1] - r[j, 1]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)

        r2_center = dx * dx + dy * dy
        if r2_center <= 1e-12:
            continue

        r_ij = np.sqrt(r2_center)
        inv_r = 1.0 / r_ij

        # Normalized direction vector
        nx = dx * inv_r
        ny = dy * inv_r

        # --- B. WCA steric repulsion ---
        sigma = 0.5 * (diameter[i] + diameter[j])
        r_cut_wca = (2.0 ** (1.0 / 6.0)) * sigma

        if r_ij < r_cut_wca:
            sig_r = sigma * inv_r
            sig2_r2 = sig_r * sig_r
            sig6_r6 = sig2_r2 * sig2_r2 * sig2_r2
            f_mag_wca = 48.0 * epsilon_wca * inv_r * sig6_r6 * (sig6_r6 - 0.5)

            # Cap force magnitude for numerical stability
            if f_mag_wca > 1000.0:
                f_mag_wca = 1000.0

            fx = f_mag_wca * nx
            fy = f_mag_wca * ny
            F[i, 0] += fx
            F[i, 1] += fy
            F[j, 0] -= fx
            F[j, 1] -= fy

        # --- C. Active non-reciprocal interaction (skin distance) ---
        if r2_center > cut2_nr:
            continue

        skin_distance = r_ij - sigma
        if skin_distance < SATURATION_DISTANCE:
            constrained_dist = SATURATION_DISTANCE
        else:
            constrained_dist = skin_distance

        dist_sq = constrained_dist * constrained_dist
        factor = ag2 / dist_sq

        # Case 1: i acts as the source
        if is_active[i] and in_wave[i]:
            mag = eps_dark if wave_type[i] == 0 else -eps_bright
            F[j, 0] += mag * factor * nx
            F[j, 1] += mag * factor * ny

        # Case 2: j acts as the source
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

        # --- Basic parameters ---
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

        # --- Traveling wave parameters ---
        self.wave_speed = wave_speed
        self.wavelength = wavelength
        self.dark_width = (1.0 - duty_cycle) * self.wavelength
        self.cut_nr = cut_nr
        self.verlet_skin = verlet_skin

        # --- Non-reciprocal interaction strengths ---
        self.eps_dark = eps_dark
        self.eps_bright = beta * eps_dark

        self.print_every_steps = max(1, int(print_every_time / dt))
        self.rng = np.random.default_rng(seed)

        # --- Output ---
        self.output_dir = output_dir
        self.save_snapshots = save_snapshots
        if self.save_snapshots:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        # --- State arrays ---
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
        self.cutoff_global = max((2.0 ** (1.0 / 6.0)) * max_sigma, self.cut_nr)
        self.cutoff_verlet = self.cutoff_global + self.verlet_skin
        self._rebuild_verlet()

        actual_area = self.Lx * self.Ly
        particle_area = self.n_active * np.pi * (self.ag_diameter / 2) ** 2
        print(f"Actual area fraction: {particle_area / actual_area:.4f}")

    def _init_particles(self):
        """
        Improved initialization:
        1. Larger particles are placed first (sorted by diameter).
        2. Up to 100,000 placement attempts per particle.
        3. Distances are checked using periodic boundary conditions.
        """
        self.diameter[:self.n_active] = self.ag_diameter
        self.diameter[self.n_active:] = self.sio2_diameter
        self.is_active[:self.n_active] = True
        self.is_tracer[self.n_active:] = True

        sorted_indices = np.argsort(self.diameter)[::-1]
        placed_mask = np.zeros(self.N, dtype=np.bool_)

        for i in sorted_indices:
            placed = False
            curr_diam = self.diameter[i]

            for attempt in range(100000):
                x = self.rng.random() * self.Lx
                y = self.rng.random() * self.Ly

                if not np.any(placed_mask):
                    self.r[i] = (x, y)
                    placed = True
                    break

                existing = np.where(placed_mask)[0]
                dx = self.r[existing, 0] - x
                dy = self.r[existing, 1] - y
                dx -= self.Lx * np.round(dx / self.Lx)
                dy -= self.Ly * np.round(dy / self.Ly)

                dist2 = dx * dx + dy * dy
                min_dist = 0.5 * (self.diameter[existing] + curr_diam)
                if np.all(dist2 >= min_dist * min_dist):
                    self.r[i] = (x, y)
                    placed = True
                    break

            if not placed:
                print(f"Warning: Particle {i} could not be placed without overlap. Forced placement.")
                self.r[i] = (self.rng.random() * self.Lx, self.rng.random() * self.Ly)

            placed_mask[i] = True

    def _init_diffusion(self):
        radius = 0.5 * self.diameter
        self.D = self.kBT / (6.0 * np.pi * self.eta * radius)
        self.gamma = self.kBT / self.D
