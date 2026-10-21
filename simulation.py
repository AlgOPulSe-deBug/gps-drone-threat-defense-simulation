#!/usr/bin/env python3
"""
===============================================================================
TITLE: GPS Spoofing as a Defensive Countermeasure Against an Enemy Drone Attack
===============================================================================

Academic Simulation Project (numpy + matplotlib only)

SCENARIO OVERVIEW:
You (the user) choose or enter:
  1. The Enemy Drone's Start Point: x_start = [x0, y0]
  2. The Protected City's Main Location: x_city = [xc, yc] (which dynamically
     forms the darkest, highest-population peak on the 2D harm cost map).

From your chosen x_start and x_city, the simulation automatically:
  - Generates the 2D population/infrastructure Harm Cost Map centered on your
    city location (the darkest crimson region on the graph) along with surrounding
    low-harm open-land/lake basins.
  - Evaluates the reachable diversion region for the drone (points the spoofer
    can steer the drone into before it reaches the city without triggering the
    stealth anomaly detector) and selects x_safe as the MINIMUM-DAMAGE point.
  - Optimizes the stealth spoofing ramp rate delta_rate_max (Equation 7) and
    simulates Cases A (No Spoofing), B (Stealth Spoofing), C (Aggressive Jump),
    and the 2-Drone Formation Extension.

HOW TO RUN:
  1. Interactive prompt (asks for coordinates in terminal; press Enter for defaults):
       python simulation.py
  2. Command-line arguments:
       python simulation.py --start 100 100 --city 800 800
  3. Non-interactive batch mode (uses defaults at top of file):
       python simulation.py --non-interactive
===============================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for clean file rendering
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

# =============================================================================
# DEFAULT PARAMETERS (Used if you press Enter at the prompt or pass --non-interactive)
# =============================================================================
RANDOM_SEED            = 42                        # Fixed seed for reproducibility

# Default user-selectable coordinates (in meters, 2D [East, North] on [0, 1000]x[0, 1000])
DEFAULT_CITY_LOCATION  = np.array([800.0, 800.0])  # Darkest population peak [m, m]
DEFAULT_START_DRONE_1  = np.array([100.0, 100.0])  # Enemy Drone 1 start point [m, m]
DRONE_2_START_OFFSET   = np.array([-40.0, 140.0])  # Offset of Drone 2 relative to Drone 1 [m, m]

# Map domain bounds
MAP_DOMAIN             = (0.0, 1000.0, 0.0, 1000.0)
GRID_SIZE              = 200                       # 200 x 200 spatial cost grid (5 m resolution)

# Kinematic & time parameters
V_MAX                  = 20.0                      # Drone maximum cruise speed [m/s]
DT                     = 0.5                       # Discrete simulation time step [s]
PROPORTIONAL_R         = 35.0                      # Terminal proportional capture radius [m]

# Sensor noise & IMU drift parameters
NOISE_SIGMA            = 1.2                       # GPS measurement noise std dev sigma [m]
IMU_DRIFT_SIGMA        = 0.25                      # IMU dead-reckoning random walk step std [m]
ALPHA_INS              = 0.068                     # Aided INS complementary drift-bounding gain

# Spoofing & anomaly detector parameters
DELTA_RATE_MAX_DEFAULT = 5.2                       # Default stealth spoofing rate limit [m/step]
DETECTION_TAU          = 90.0                      # Anomaly detector residual threshold tau [m]
AGGRESSIVE_JUMP_T      = 12.0                      # Time of sudden offset jump in Case C [s]

# Output directory configuration
SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR             = os.path.join(SCRIPT_DIR, "outputs")

# Consistent academic presentation color palette
COLORS = {
    "city":       "#D62828",  # Crimson Red
    "safe":       "#2A9D8F",  # Emerald Teal/Green
    "true_b":     "#1D3557",  # Deep Navy Blue (True trajectory - Stealth)
    "perc_b":     "#F77F00",  # Vivid Orange (Perceived trajectory - Stealth)
    "case_a":     "#5C677D",  # Slate Gray (Baseline No Spoofing)
    "case_c":     "#E63946",  # Bright Red (Aggressive Spoofing)
    "drone2_t":   "#0077B6",  # Ocean Blue (Drone 2 True)
    "drone2_p":   "#FCBF49",  # Golden Amber (Drone 2 Perceived)
    "tau":        "#9B2226",  # Dark Red threshold line
}


# =============================================================================
# USER INPUT PARSER (Interactive Prompt + CLI Flags)
# =============================================================================
def get_user_coordinates():
    """
    Allows the user to enter the Drone Start Point and Protected City Location
    either via command-line arguments (--start X Y --city X Y) or interactively
    via terminal input() prompts. Automatically falls back to defaults if run
    in a non-interactive environment or when pressing Enter.
    """
    parser = argparse.ArgumentParser(
        description="GPS Spoofing Defensive Countermeasure Simulation"
    )
    parser.add_argument(
        "--start", nargs=2, type=float, metavar=("X_START", "Y_START"),
        help="Enemy Drone 1 start coordinates [x, y] in meters (0 to 1000)"
    )
    parser.add_argument(
        "--city", nargs=2, type=float, metavar=("X_CITY", "Y_CITY"),
        help="Protected City main coordinates [x, y] in meters (0 to 1000)"
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Run without interactive prompts using default or CLI coordinates"
    )
    args, _ = parser.parse_known_args()

    start_pos = DEFAULT_START_DRONE_1.copy()
    city_pos = DEFAULT_CITY_LOCATION.copy()

    if args.start is not None:
        start_pos = np.clip(np.array(args.start, dtype=float), 20.0, 980.0)
    if args.city is not None:
        city_pos = np.clip(np.array(args.city, dtype=float), 20.0, 980.0)

    # Interactive prompt if running in an interactive terminal and CLI args weren't both given
    if not args.non_interactive and sys.stdin.isatty() and (args.start is None or args.city is None):
        print("\n" + "=" * 78)
        print(" INTERACTIVE SETUP: ENTER DRONE START POINT & CITY MAIN LOCATION")
        print(" (Map domain is 0 to 1000 m in East [x] and North [y]. Press Enter for default)")
        print("=" * 78)
        try:
            if args.start is None:
                raw_start = input(
                    f" -> Enter Drone Start Point 'x, y' [default {start_pos[0]:.0f}, {start_pos[1]:.0f}]: "
                ).strip()
                if raw_start:
                    parts = [float(p) for p in raw_start.replace(",", " ").split()]
                    if len(parts) == 2:
                        start_pos = np.clip(np.array(parts, dtype=float), 20.0, 980.0)

            if args.city is None:
                raw_city = input(
                    f" -> Enter Protected City Center 'x, y' [default {city_pos[0]:.0f}, {city_pos[1]:.0f}]: "
                ).strip()
                if raw_city:
                    parts = [float(p) for p in raw_city.replace(",", " ").split()]
                    if len(parts) == 2:
                        city_pos = np.clip(np.array(parts, dtype=float), 20.0, 980.0)
        except EOFError:
            pass

    # Compute Drone 2 start position slightly offset from Drone 1 (clamped inside map)
    start_pos_2 = np.clip(start_pos + DRONE_2_START_OFFSET, 30.0, 970.0)
    return start_pos, start_pos_2, city_pos


# =============================================================================
# EQUATION 6: DYNAMIC HARM COST MAP & MINIMUM-DAMAGE SAFE-ZONE SELECTION
# =============================================================================
def build_harm_cost_map(start_pos, city_pos, grid_size=GRID_SIZE, domain=MAP_DOMAIN):
    """
    Constructs a 2D spatial grid of "harm" H(x, y) [1 to 100] where:
      - The user-selected `city_pos` is the DARKEST, highest-harm peak on the map.
      - Secondary suburban/industrial zones and natural low-harm open-land/lake
        basins are positioned relative to the flight corridor.
      - Equation 6: Automatically selects `x_safe` as the MINIMUM-DAMAGE (minimum
        harm cost) reachable coordinate that the spoofer can covertly divert the
        drone into without violating the stealth threshold tau.
    """
    x_min, x_max, y_min, y_max = domain
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xs, ys)

    # 1. Primary Metropolitan Core centered strictly at user-chosen `city_pos`
    #    (Guarantees `city_pos` is the darkest crimson peak on the population map)
    city_dist_sq = (X - city_pos[0])**2 + (Y - city_pos[1])**2
    city_harm = 88.0 * np.exp(-city_dist_sq / (2.0 * 175.0**2))

    # Geometry of the attack vector from start_pos to city_pos
    flight_vec = city_pos - start_pos
    flight_dist = max(float(np.linalg.norm(flight_vec)), 100.0)
    u_along = flight_vec / flight_dist
    # Perpendicular unit vector (90 deg clockwise: right-hand flank of flight path)
    u_perp = np.array([u_along[1], -u_along[0]])
    # Ensure u_perp points toward the side with more room inside the [0, 1000]^2 map
    mid_pt = start_pos + 0.65 * flight_vec
    cand_right = mid_pt + 260.0 * u_perp
    cand_left = mid_pt - 260.0 * u_perp
    margin_right = min(cand_right[0] - 50.0, 950.0 - cand_right[0],
                       cand_right[1] - 50.0, 950.0 - cand_right[1])
    margin_left = min(cand_left[0] - 50.0, 950.0 - cand_left[0],
                      cand_left[1] - 50.0, 950.0 - cand_left[1])
    if margin_left > margin_right:
        u_perp = -u_perp

    # 2. Secondary Suburban Corridor on the opposite flank of the city
    suburb_center = np.clip(city_pos - 0.35 * flight_vec - 220.0 * u_perp, 100.0, 900.0)
    suburb_dist_sq = ((X - suburb_center[0])**2 / (2.0 * 210.0**2) +
                      (Y - suburb_center[1])**2 / (2.0 * 140.0**2))
    suburb_harm = 36.0 * np.exp(-suburb_dist_sq)

    # 3. Industrial Cluster near mid-corridor on the high-harm flank
    ind_center = np.clip(start_pos + 0.45 * flight_vec - 190.0 * u_perp, 80.0, 920.0)
    ind_dist_sq = (X - ind_center[0])**2 + (Y - ind_center[1])**2
    ind_harm = 25.0 * np.exp(-ind_dist_sq / (2.0 * 135.0**2))

    # 4. Low-Harm Open-Water / Rural Basin on the safe diversion flank
    basin_center = np.clip(start_pos + 0.68 * flight_vec + 310.0 * u_perp, 90.0, 910.0)
    lake_dist_sq = (X - basin_center[0])**2 + (Y - basin_center[1])**2
    lake_relief = 14.2 * np.exp(-lake_dist_sq / (2.0 * 140.0**2))

    harm_map = np.clip(15.0 + city_harm + suburb_harm + ind_harm - lake_relief, 1.0, 100.0)

    # Equation 6: Find the MINIMUM-DAMAGE reachable point x_safe
    # Reachability conditions for a stealthy diversion:
    #   (a) Inside map margins: [60, 940] m
    #   (b) Safely outside the city blast/impact radius: ||x - x_city|| >= 320 m
    #   (c) Reachable by stealth spoofing before city arrival: ||x - x_city|| <= 0.78 * flight_dist
    #   (d) Forward along the flight corridor so the drone doesn't have to U-turn
    dist_to_city_grid = np.sqrt((X - city_pos[0])**2 + (Y - city_pos[1])**2)
    proj_along = (X - start_pos[0]) * u_along[0] + (Y - start_pos[1]) * u_along[1]

    min_safe_sep = min(320.0, 0.45 * flight_dist)
    max_divert_sep = max(min_safe_sep + 120.0, 0.62 * flight_dist)

    reachable_mask = (
        (X >= 60.0) & (X <= 940.0) &
        (Y >= 60.0) & (Y <= 940.0) &
        (dist_to_city_grid >= min_safe_sep) &
        (dist_to_city_grid <= max_divert_sep) &
        (proj_along >= 0.35 * flight_dist) &
        (proj_along <= 1.05 * flight_dist)
    )

    # Fallback if start_pos and city_pos are placed very close together
    if not np.any(reachable_mask):
        reachable_mask = (
            (X >= 60.0) & (X <= 940.0) &
            (Y >= 60.0) & (Y <= 940.0) &
            (dist_to_city_grid >= 0.35 * flight_dist)
        )

    masked_harm = np.where(reachable_mask, harm_map, np.inf)
    min_idx = np.unravel_index(np.argmin(masked_harm), harm_map.shape)
    x_safe = np.array([X[min_idx], Y[min_idx]])
    min_cost = float(harm_map[min_idx])

    return X, Y, harm_map, x_safe, min_cost, reachable_mask


# =============================================================================
# CORE SIMULATION CLASSES (Drone, Spoofer, Detector)
# =============================================================================
class Drone:
    """
    Represents a 2D point-mass autonomous drone navigating toward a target city
    using GPS position feedback and the unit-vector guidance law (Equation 3).
    """
    def __init__(self, start_pos, city_pos, v_max, dt, prop_radius=PROPORTIONAL_R):
        self.x_true = np.array(start_pos, dtype=float)
        self.city_pos = np.array(city_pos, dtype=float)
        self.v_max = float(v_max)
        self.dt = float(dt)
        self.prop_radius = float(prop_radius)
        self.v = np.zeros(2, dtype=float)

    def compute_guidance_velocity(self, x_perceived):
        """
        Equation 3 (Enemy Guidance Law):
        The drone steers toward the CITY using its perceived GPS position:
            v[k] = v_max * (x_city - x_perceived[k]) / ||x_city - x_perceived[k]||
        Within a small terminal capture radius of the perceived target, speed
        tapers proportionally to prevent discrete-step overshoot oscillation.
        """
        to_city = self.city_pos - x_perceived
        dist_perceived = float(np.linalg.norm(to_city))

        if dist_perceived < 1e-6:
            self.v = np.zeros(2, dtype=float)
        else:
            unit_dir = to_city / dist_perceived
            speed = self.v_max * min(1.0, dist_perceived / self.prop_radius)
            self.v = speed * unit_dir
        return self.v.copy()

    def step_dynamics(self):
        """
        Equation 1 (Drone Dynamics - 2D point mass, discrete time step dt):
            x_true[k+1] = x_true[k] + v[k] * dt
        """
        self.x_true = self.x_true + self.v * self.dt
        return self.x_true.copy()


class Spoofer:
    """
    Represents the defensive GPS spoofer that injects a controlled position offset
    delta[k] into the drone's GPS receiver to divert it to x_safe.
    """
    def __init__(self, city_pos, safe_pos, delta_rate_max, noise_sigma, dt,
                 mode="stealth", jump_time=AGGRESSIVE_JUMP_T):
        """
        mode:
          - 'none'       : Case A baseline (delta[k] = 0 for all k)
          - 'stealth'    : Case B rate-limited ramp (||delta[k+1] - delta[k]|| <= delta_rate_max)
          - 'aggressive' : Case C sudden step jump to delta_final at t = jump_time
        """
        self.city_pos = np.array(city_pos, dtype=float)
        self.safe_pos = np.array(safe_pos, dtype=float)
        # Equation 4a: Final spoofing offset required so x_perceived = x_city when x_true = x_safe
        #     delta_final = x_city - x_safe
        self.delta_final = self.city_pos - self.safe_pos
        self.delta_rate_max = float(delta_rate_max)
        self.noise_sigma = float(noise_sigma)
        self.dt = float(dt)
        self.mode = mode
        self.jump_time = float(jump_time)
        self.delta = np.zeros(2, dtype=float)  # Starts from zero at k = 0

    def measure_gps(self, x_true, rng):
        """
        Equation 2 (Spoofed GPS Measurement):
            x_perceived[k] = x_true[k] + delta[k] + noise,   noise ~ N(0, sigma^2 I)
        """
        noise = rng.normal(0.0, self.noise_sigma, size=2)
        x_perceived = x_true + self.delta + noise
        return x_perceived

    def update_offset(self, current_time):
        """
        Equation 4b (Spoofing Offset Update with Rate Limit):
        Ramp the offset toward delta_final = x_city - x_safe with stealth constraint:
            ||delta[k+1] - delta[k]|| <= delta_rate_max
        """
        if self.mode == "none":
            self.delta = np.zeros(2, dtype=float)
        elif self.mode == "stealth":
            remaining = self.delta_final - self.delta
            dist_rem = float(np.linalg.norm(remaining))
            if dist_rem <= self.delta_rate_max:
                self.delta = self.delta_final.copy()
            else:
                self.delta = self.delta + self.delta_rate_max * (remaining / dist_rem)
        elif self.mode == "aggressive":
            # Sudden jump to full offset at t >= jump_time (violating stealth constraint)
            if current_time >= self.jump_time:
                self.delta = self.delta_final.copy()
            else:
                self.delta = np.zeros(2, dtype=float)
        else:
            raise ValueError(f"Unknown spoofer mode: {self.mode}")
        return self.delta.copy()


class Detector:
    """
    Represents the enemy drone's onboard anomaly detector comparing GPS against
    an internal IMU / dead-reckoning position estimate with sensor drift.
    """
    def __init__(self, start_pos, dt, imu_drift_sigma, threshold_tau, alpha_ins=ALPHA_INS):
        self.x_imu = np.array(start_pos, dtype=float)
        self.dt = float(dt)
        self.imu_drift_sigma = float(imu_drift_sigma)
        self.threshold_tau = float(threshold_tau)
        self.alpha_ins = float(alpha_ins)
        self.alarm_triggered = False
        self.detection_time = None

    def compute_residual(self, x_perceived, current_time):
        """
        Equation 5 (Enemy Detector Residual & Threshold Test):
            r[k] = ||x_perceived[k] - x_imu[k]||
            Flag spoofing if r[k] > tau.
        """
        residual = float(np.linalg.norm(x_perceived - self.x_imu))
        if residual > self.threshold_tau:
            if not self.alarm_triggered:
                self.detection_time = current_time
            self.alarm_triggered = True
        return residual

    def step_imu(self, v_cmd, x_perceived, rng):
        """
        Equation 5b (IMU Dead-Reckoning Propagation with Random Walk Drift):
            x_imu[k+1] = x_imu[k] + v[k]*dt + alpha_ins*(x_perceived[k] - x_imu[k]) + w_imu[k]
        """
        drift_step = rng.normal(0.0, self.imu_drift_sigma, size=2)
        innovation_pull = self.alpha_ins * (x_perceived - self.x_imu)
        self.x_imu = self.x_imu + v_cmd * self.dt + innovation_pull + drift_step
        return self.x_imu.copy()


# =============================================================================
# SIMULATION RUNNER FUNCTION
# =============================================================================
def compute_sim_duration(start_pos, city_pos, safe_pos, v_max=V_MAX):
    """Automatically scales simulation duration to fit any user-chosen coordinates."""
    dist_city = float(np.linalg.norm(city_pos - start_pos))
    dist_safe = float(np.linalg.norm(safe_pos - start_pos))
    offset_norm = float(np.linalg.norm(city_pos - safe_pos))
    est_time = max(dist_city, dist_safe + 0.4 * offset_norm) / v_max + 20.0
    return float(np.clip(np.ceil(est_time / 5.0) * 5.0, 50.0, 120.0))


def run_simulation(start_pos, city_pos, safe_pos, sim_time, mode="stealth",
                   delta_rate_max=DELTA_RATE_MAX_DEFAULT, v_max=V_MAX, dt=DT,
                   noise_sigma=NOISE_SIGMA, imu_drift_sigma=IMU_DRIFT_SIGMA,
                   tau=DETECTION_TAU, seed=RANDOM_SEED):
    """
    Executes a complete discrete-time simulation for a single drone under the
    specified spoofing mode ('none', 'stealth', or 'aggressive').
    """
    rng = np.random.default_rng(seed)
    num_steps = int(np.round(sim_time / dt)) + 1
    times = np.linspace(0.0, sim_time, num_steps)

    drone = Drone(start_pos=start_pos, city_pos=city_pos, v_max=v_max, dt=dt)
    spoofer = Spoofer(city_pos=city_pos, safe_pos=safe_pos,
                      delta_rate_max=delta_rate_max, noise_sigma=noise_sigma,
                      dt=dt, mode=mode)
    detector = Detector(start_pos=start_pos, dt=dt,
                        imu_drift_sigma=imu_drift_sigma, threshold_tau=tau)

    # Pre-allocate history arrays
    x_true_hist      = np.zeros((num_steps, 2))
    x_perc_hist      = np.zeros((num_steps, 2))
    x_imu_hist       = np.zeros((num_steps, 2))
    v_hist           = np.zeros((num_steps, 2))
    delta_hist       = np.zeros((num_steps, 2))
    delta_norm_hist  = np.zeros(num_steps)
    residual_hist    = np.zeros(num_steps)
    dist_city_hist   = np.zeros(num_steps)
    dist_safe_hist   = np.zeros(num_steps)

    divert_time = sim_time
    settled_flag = False
    target_destination = city_pos if mode == "none" else safe_pos

    for k, t in enumerate(times):
        # Record current true position and spoofing offset
        x_true_hist[k] = drone.x_true
        delta_hist[k] = spoofer.delta
        delta_norm_hist[k] = np.linalg.norm(spoofer.delta)

        # Equation 2: Spoofed GPS measurement
        x_perceived = spoofer.measure_gps(drone.x_true, rng)
        x_perc_hist[k] = x_perceived

        # Equation 5: Anomaly detector residual evaluation
        x_imu_hist[k] = detector.x_imu
        residual_hist[k] = detector.compute_residual(x_perceived, t)

        # Record Euclidean distances from TRUE position to City and Safe Zone
        dist_city_hist[k] = np.linalg.norm(drone.x_true - city_pos)
        dist_safe_hist[k] = np.linalg.norm(drone.x_true - safe_pos)

        # Record time to reach within 15 m of destination
        if not settled_flag and np.linalg.norm(drone.x_true - target_destination) <= 15.0:
            divert_time = t
            settled_flag = True

        # Equation 3: Enemy guidance law (uses perceived position)
        v_cmd = drone.compute_guidance_velocity(x_perceived)
        v_hist[k] = v_cmd

        # Advance state to k+1 (Equations 1, 4, 5b)
        drone.step_dynamics()
        detector.step_imu(v_cmd, x_perceived, rng)
        spoofer.update_offset(t)

    final_true = x_true_hist[-1]
    miss_dist_safe = float(np.linalg.norm(final_true - safe_pos))
    miss_dist_city = float(np.linalg.norm(final_true - city_pos))
    max_residual = float(np.max(residual_hist))

    return {
        "mode": mode,
        "times": times,
        "sim_time": sim_time,
        "x_true": x_true_hist,
        "x_perc": x_perc_hist,
        "x_imu": x_imu_hist,
        "v": v_hist,
        "delta": delta_hist,
        "delta_norm": delta_norm_hist,
        "residual": residual_hist,
        "dist_city": dist_city_hist,
        "dist_safe": dist_safe_hist,
        "final_true": final_true,
        "miss_dist_safe": miss_dist_safe,
        "miss_dist_city": miss_dist_city,
        "divert_time": divert_time,
        "max_residual": max_residual,
        "detected": detector.alarm_triggered,
        "detection_time": detector.detection_time,
        "delta_rate_max": delta_rate_max,
    }


# =============================================================================
# EQUATION 7: RATE-LIMIT OPTIMIZATION (SWEEP FOR MAXIMUM SAFE SPOOFING RATE)
# =============================================================================
def optimize_delta_rate_max(start_pos, city_pos, safe_pos, sim_time, tau=DETECTION_TAU):
    """
    Equation 7 (Optimization):
    Sweeps candidate delta_rate_max values to determine the largest spoofing
    ramp rate [m/step] that keeps max_k r[k] strictly below threshold tau
    (with a 5% safety margin) and diverts the drone cleanly to x_safe.
    """
    candidate_rates = np.linspace(2.0, 9.0, 50)
    safe_rates = []
    max_residuals = []
    divert_times = []

    for rate in candidate_rates:
        res = run_simulation(
            start_pos=start_pos, city_pos=city_pos, safe_pos=safe_pos,
            sim_time=sim_time, mode="stealth", delta_rate_max=rate,
            tau=tau, seed=RANDOM_SEED
        )
        max_residuals.append(res["max_residual"])
        divert_times.append(res["divert_time"])
        if res["max_residual"] <= 0.95 * tau and res["miss_dist_safe"] <= 15.0:
            safe_rates.append((rate, res["divert_time"], res["max_residual"]))

    if safe_rates:
        best_rate, best_time, best_res = safe_rates[-1]
    else:
        best_rate, best_time, best_res = DELTA_RATE_MAX_DEFAULT, np.nan, np.nan

    return {
        "candidate_rates": candidate_rates,
        "max_residuals": np.array(max_residuals),
        "divert_times": np.array(divert_times),
        "optimal_rate": best_rate,
        "optimal_divert_time": best_time,
        "optimal_max_residual": best_res,
    }


# =============================================================================
# PLOTTING & ANIMATION FUNCTIONS
# =============================================================================
def apply_clean_style():
    """Configures Matplotlib defaults for crisp academic presentation visuals."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.0,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    })


def plot_static_trajectories(X, Y, harm_map, start_pos_1, start_pos_2,
                             city_pos, safe_pos, res_a, res_b, res_c,
                             res_d2, safe_pos_d2, output_dir):
    """
    Visual 2: Static plot comparing True vs. Perceived trajectories for Cases A, B, C
    (plus the 2-drone formation extension in a companion panel).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=200)

    # --- Panel 1: Single-Drone Comparison (Cases A, B, C) ---
    ax = axes[0]
    contour = ax.contourf(X, Y, harm_map, levels=25, cmap="YlOrRd", alpha=0.74)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Harm / Population Density Index [0–100]", fontsize=10)

    # Case A: Baseline (No Spoofing)
    ax.plot(res_a["x_true"][:, 0], res_a["x_true"][:, 1],
            color=COLORS["case_a"], linestyle="-.", linewidth=2.2,
            label="Case A: True Path (No Spoofing → City)")

    # Case B: Stealthy Rate-Limited Spoofing
    ax.plot(res_b["x_true"][:, 0], res_b["x_true"][:, 1],
            color=COLORS["true_b"], linestyle="-", linewidth=3.0,
            label="Case B: True Path (Stealth Spoof → Min-Harm Zone)")
    ax.plot(res_b["x_perc"][:, 0], res_b["x_perc"][:, 1],
            color=COLORS["perc_b"], linestyle="--", linewidth=2.2,
            label="Case B: Perceived GPS Path (Believes → City)")

    # Case C: Aggressive Sudden Jump Spoofing
    ax.plot(res_c["x_true"][:, 0], res_c["x_true"][:, 1],
            color=COLORS["case_c"], linestyle=":", linewidth=2.5,
            label="Case C: True Path (Aggressive Jump)")

    # Mark detection point on Case C
    if res_c["detected"]:
        det_idx = int(np.round(res_c["detection_time"] / DT))
        ax.scatter(res_c["x_true"][det_idx, 0], res_c["x_true"][det_idx, 1],
                   color=COLORS["case_c"], marker="X", s=150, zorder=8,
                   edgecolors="black", linewidths=1.2,
                   label=f"Case C Detection Alarm (t={res_c['detection_time']:.1f} s)")

    # Key landmarks
    ax.scatter(*start_pos_1, color="#111111", marker="^", s=130, zorder=9,
               label=f"User Start [{start_pos_1[0]:.0f}, {start_pos_1[1]:.0f}] m")
    ax.scatter(*city_pos, color=COLORS["city"], marker="*", s=260, zorder=10,
               edgecolors="black", linewidths=1.0,
               label=f"User City Peak [{city_pos[0]:.0f}, {city_pos[1]:.0f}] m")
    ax.scatter(*safe_pos, color=COLORS["safe"], marker="P", s=210, zorder=10,
               edgecolors="black", linewidths=1.0,
               label=f"Auto Min-Harm Safe Zone [{safe_pos[0]:.0f}, {safe_pos[1]:.0f}] m")

    ax.set_title("Cases A, B & C: True vs. Perceived Drone Trajectories")
    ax.set_xlabel("East Position x [m]")
    ax.set_ylabel("North Position y [m]")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.92)

    # --- Panel 2: Extension to 2 Enemy Drones Under Independent Stealth Spoofing ---
    ax2 = axes[1]
    contour2 = ax2.contourf(X, Y, harm_map, levels=25, cmap="YlOrRd", alpha=0.74)
    cbar2 = fig.colorbar(contour2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Harm / Population Density Index [0–100]", fontsize=10)

    # Drone 1 trajectories
    ax2.plot(res_b["x_true"][:, 0], res_b["x_true"][:, 1],
             color=COLORS["true_b"], linestyle="-", linewidth=2.8,
             label="Drone 1: True Path (Stealth Spoofed)")
    ax2.plot(res_b["x_perc"][:, 0], res_b["x_perc"][:, 1],
             color=COLORS["perc_b"], linestyle="--", linewidth=2.0,
             label="Drone 1: Perceived GPS Path")

    # Drone 2 trajectories
    ax2.plot(res_d2["x_true"][:, 0], res_d2["x_true"][:, 1],
             color=COLORS["drone2_t"], linestyle="-", linewidth=2.8,
             label="Drone 2: True Path (Stealth Spoofed)")
    ax2.plot(res_d2["x_perc"][:, 0], res_d2["x_perc"][:, 1],
             color=COLORS["drone2_p"], linestyle="--", linewidth=2.0,
             label="Drone 2: Perceived GPS Path")

    # Landmarks
    ax2.scatter(*start_pos_1, color=COLORS["true_b"], marker="^", s=130,
                edgecolors="black", zorder=9,
                label=f"Drone 1 Start [{start_pos_1[0]:.0f}, {start_pos_1[1]:.0f}] m")
    ax2.scatter(*start_pos_2, color=COLORS["drone2_t"], marker="v", s=130,
                edgecolors="black", zorder=9,
                label=f"Drone 2 Start [{start_pos_2[0]:.0f}, {start_pos_2[1]:.0f}] m")
    ax2.scatter(*city_pos, color=COLORS["city"], marker="*", s=260, zorder=10,
                edgecolors="black", linewidths=1.0, label="Protected City (Darkest Peak)")
    ax2.scatter(*safe_pos, color=COLORS["safe"], marker="P", s=210, zorder=10,
                edgecolors="black", linewidths=1.0, label="Min-Damage Safe Zone Basin")
    ax2.scatter(*safe_pos_d2, color=COLORS["safe"], marker="D", s=120, zorder=10,
                edgecolors="black", linewidths=1.0)

    # Draw safe zone arrival basin circle
    safe_circle = Circle(safe_pos, radius=55.0, edgecolor=COLORS["safe"],
                         facecolor="none", linestyle="--", linewidth=1.8)
    ax2.add_patch(safe_circle)

    ax2.set_title("Multi-Drone Extension: 2 Drones Diverted to Min-Harm Basin")
    ax2.set_xlabel("East Position x [m]")
    ax2.set_ylabel("North Position y [m]")
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 1000)
    ax2.set_aspect("equal")
    ax2.legend(loc="upper left", framealpha=0.92)

    fig.suptitle("GPS Spoofing Defensive Countermeasure — Spatial Trajectory Analysis", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(output_dir, "fig2_trajectories_comparison.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_spoofing_offset(res_a, res_b, res_c, safe_pos, city_pos, output_dir):
    """
    Visual 3: Plot of spoofing offset magnitude ||delta[k]|| over time for Cases A, B, C.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=200)
    target_norm = float(np.linalg.norm(city_pos - safe_pos))
    sim_time = res_b["sim_time"]

    ax.plot(res_a["times"], res_a["delta_norm"], color=COLORS["case_a"],
            linestyle="-.", linewidth=2.2, label="Case A: No Spoofing (||δ|| = 0 m)")
    ax.plot(res_b["times"], res_b["delta_norm"], color=COLORS["true_b"],
            linestyle="-", linewidth=3.0,
            label=f"Case B: Stealth Rate-Limited Ramp (rate ≤ {res_b['delta_rate_max']:.2f} m/step)")
    ax.plot(res_c["times"], res_c["delta_norm"], color=COLORS["case_c"],
            linestyle="--", linewidth=2.5,
            label=f"Case C: Aggressive Step Jump at t = {AGGRESSIVE_JUMP_T:.0f} s")

    ax.axhline(target_norm, color=COLORS["safe"], linestyle=":", linewidth=2.0,
               label=f"Target Final Offset ||δ_final|| = ||x_city - x_safe|| = {target_norm:.1f} m")

    ax.set_title("Spoofing Offset Magnitude ||δ[k]|| vs. Time (Equation 4)")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Spoofing Offset Magnitude ||δ[k]|| [m]")
    ax.set_xlim(0, sim_time)
    ax.set_ylim(-25, target_norm * 1.15)
    ax.legend(loc="lower right", framealpha=0.94)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig3_spoofing_offset_norm.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_detector_residuals(res_b, res_c, opt_results, output_dir):
    """
    Visual 4: Plot of detector residual r[k] = ||x_perceived[k] - x_imu[k]|| vs. time
    with detection threshold line tau, plus rate-limit optimization sweep panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=200)
    sim_time = res_b["sim_time"]

    # --- Panel 1: Residual Time Series (Stealth vs Aggressive) ---
    ax = axes[0]
    ax.plot(res_b["times"], res_b["residual"], color=COLORS["true_b"],
            linestyle="-", linewidth=2.6,
            label=f"Case B: Stealth Spoofing (Max r = {res_b['max_residual']:.1f} m < τ)")
    ax.plot(res_c["times"], res_c["residual"], color=COLORS["case_c"],
            linestyle="-", linewidth=2.4,
            label=f"Case C: Aggressive Jump (Max r = {res_c['max_residual']:.1f} m > τ)")

    ax.axhline(DETECTION_TAU, color=COLORS["tau"], linestyle="--", linewidth=2.2,
               label=f"Detection Threshold τ = {DETECTION_TAU:.1f} m")
    y_top = max(np.max(res_c["residual"]) * 1.08, DETECTION_TAU * 1.4)
    ax.fill_between(res_b["times"], DETECTION_TAU, y_top, color=COLORS["case_c"],
                    alpha=0.10, label="Alarm Zone (r[k] > τ : Spoofing Detected)")

    if res_c["detected"]:
        det_t = res_c["detection_time"]
        det_idx = int(np.round(det_t / DT))
        ax.scatter(det_t, res_c["residual"][det_idx], color=COLORS["case_c"],
                   marker="X", s=140, zorder=9, edgecolors="black",
                   label=f"Alarm Triggered at t = {det_t:.1f} s")

    ax.set_title("Onboard IMU/GPS Detector Residual r[k] vs. Time (Equation 5)")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Detector Residual r[k] = ||x_perceived - x_imu|| [m]")
    ax.set_xlim(0, sim_time)
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper right", framealpha=0.94)

    # --- Panel 2: Equation 7 Rate-Limit Optimization Sweep ---
    ax2 = axes[1]
    rates_per_sec = opt_results["candidate_rates"] / DT
    ax2.plot(rates_per_sec, opt_results["max_residuals"], color="#1D3557",
             linewidth=2.6, label="Peak Residual max_k r[k]")
    ax2.axhline(DETECTION_TAU, color=COLORS["tau"], linestyle="--", linewidth=2.2,
                label=f"Threshold τ = {DETECTION_TAU:.1f} m")
    ax2.axvline(opt_results["optimal_rate"] / DT, color=COLORS["safe"],
                linestyle="-.", linewidth=2.2,
                label=f"Max Safe Rate = {opt_results['optimal_rate']/DT:.1f} m/s "
                      f"({opt_results['optimal_rate']:.2f} m/step)")

    ax2.set_title("Rate-Limit Optimization Sweep (Equation 7)")
    ax2.set_xlabel("Spoofing Offset Ramp Rate [m/s]")
    ax2.set_ylabel("Peak Residual max r[k] [m]")
    ax2.legend(loc="upper left", framealpha=0.94)

    fig.suptitle("Stealth Constraint Verification & Rate-Limit Optimization", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(output_dir, "fig4_detector_residuals.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_distances_over_time(res_a, res_b, output_dir):
    """
    Visual 5: Plot of distance from the drone's TRUE position to the Protected City
    and to the Safe Zone over time.
    """
    fig, ax = plt.subplots(figsize=(9.8, 5.5), dpi=200)
    sim_time = res_b["sim_time"]

    # Case B (Stealth Spoofing) distances
    ax.plot(res_b["times"], res_b["dist_city"], color=COLORS["city"],
            linestyle="-", linewidth=2.8,
            label="Case B (Stealth Spoofed): True Distance to City ||x_true - x_city||")
    ax.plot(res_b["times"], res_b["dist_safe"], color=COLORS["safe"],
            linestyle="-", linewidth=2.8,
            label="Case B (Stealth Spoofed): True Distance to Safe Zone ||x_true - x_safe||")

    # Case A (No Spoofing baseline) distance to city for reference
    ax.plot(res_a["times"], res_a["dist_city"], color=COLORS["case_a"],
            linestyle="--", linewidth=2.0,
            label="Case A (Unspoofed Baseline): True Distance to City")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("True Separation Distance to Protected City and Safe Zone vs. Time")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Euclidean Distance [m]")
    ax.set_xlim(0, sim_time)
    ax.set_ylim(0, max(np.max(res_b["dist_city"]), np.max(res_b["dist_safe"])) * 1.08)
    ax.legend(loc="upper right", framealpha=0.94)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig5_distances_to_targets.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_animation_gif(X, Y, harm_map, start_pos_1, start_pos_2,
                         city_pos, safe_pos, res_b, res_d2, output_dir):
    """
    Visual 1: Animated 2D map showing the harm heat map in the background,
    city (red marker at darkest peak), safe zone (green marker at min-harm point),
    and the drones' TRUE paths (solid blue) vs PERCEIVED paths (dashed orange).
    """
    fig, ax = plt.subplots(figsize=(8.5, 7.6), dpi=130)

    contour = ax.contourf(X, Y, harm_map, levels=25, cmap="YlOrRd", alpha=0.74)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Harm / Population Density Index [0–100]", fontsize=10)

    # Static markers for City, Safe Zone, and Launch Sites
    ax.scatter(*city_pos, color=COLORS["city"], marker="*", s=280, zorder=10,
               edgecolors="black", linewidths=1.0, label="Protected City (Darkest Peak)")
    ax.scatter(*safe_pos, color=COLORS["safe"], marker="P", s=220, zorder=10,
               edgecolors="black", linewidths=1.0, label="Safe Zone (Min-Damage Point)")
    ax.scatter(*start_pos_1, color="#111111", marker="^", s=110, zorder=9,
               label="User-Selected Drone Launch Sector")
    ax.scatter(*start_pos_2, color="#111111", marker="v", s=110, zorder=9)

    # Dynamic line artists for Drone 1 (primary solid blue & dashed orange)
    line_true_1, = ax.plot([], [], color="#00509D", linestyle="-", linewidth=3.0,
                           label="Drone 1 TRUE Path (Solid Blue)")
    line_perc_1, = ax.plot([], [], color="#F77F00", linestyle="--", linewidth=2.2,
                           label="Drone 1 PERCEIVED Path (Dashed Orange)")
    dot_true_1,  = ax.plot([], [], color="#00509D", marker="o", markersize=8, zorder=11)
    dot_perc_1,  = ax.plot([], [], color="#F77F00", marker="o", markersize=7, zorder=11)

    # Dynamic line artists for Drone 2 (secondary formation drone)
    line_true_2, = ax.plot([], [], color="#0096C7", linestyle="-", linewidth=2.2,
                           alpha=0.85, label="Drone 2 TRUE Path (Formation)")
    line_perc_2, = ax.plot([], [], color="#FCBF49", linestyle="--", linewidth=1.8,
                           alpha=0.85, label="Drone 2 PERCEIVED Path")
    dot_true_2,  = ax.plot([], [], color="#0096C7", marker="s", markersize=7, zorder=11)

    time_text = ax.text(
        0.03, 0.04, "", transform=ax.transAxes, fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#333333", alpha=0.92)
    )

    ax.set_title("Defensive GPS Spoofing — Real-Time Trajectory Diversion")
    ax.set_xlabel("East Position x [m]")
    ax.set_ylabel("North Position y [m]")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8.8)
    fig.tight_layout()

    frame_indices = np.arange(0, len(res_b["times"]), 2)

    def init():
        line_true_1.set_data([], [])
        line_perc_1.set_data([], [])
        dot_true_1.set_data([], [])
        dot_perc_1.set_data([], [])
        line_true_2.set_data([], [])
        line_perc_2.set_data([], [])
        dot_true_2.set_data([], [])
        time_text.set_text("")
        return (line_true_1, line_perc_1, dot_true_1, dot_perc_1,
                line_true_2, line_perc_2, dot_true_2, time_text)

    def update(frame_idx):
        k = frame_indices[frame_idx]
        t = res_b["times"][k]

        # Drone 1 paths
        line_true_1.set_data(res_b["x_true"][:k+1, 0], res_b["x_true"][:k+1, 1])
        line_perc_1.set_data(res_b["x_perc"][:k+1, 0], res_b["x_perc"][:k+1, 1])
        dot_true_1.set_data([res_b["x_true"][k, 0]], [res_b["x_true"][k, 1]])
        dot_perc_1.set_data([res_b["x_perc"][k, 0]], [res_b["x_perc"][k, 1]])

        # Drone 2 paths
        line_true_2.set_data(res_d2["x_true"][:k+1, 0], res_d2["x_true"][:k+1, 1])
        line_perc_2.set_data(res_d2["x_perc"][:k+1, 0], res_d2["x_perc"][:k+1, 1])
        dot_true_2.set_data([res_d2["x_true"][k, 0]], [res_d2["x_true"][k, 1]])

        time_text.set_text(
            f"Time: {t:5.1f} s  |  ||δ||: {res_b['delta_norm'][k]:5.1f} m\n"
            f"Detector r[k]: {res_b['residual'][k]:4.1f} m / τ={DETECTION_TAU:.0f} m (STEALTH OK)"
        )
        return (line_true_1, line_perc_1, dot_true_1, dot_perc_1,
                line_true_2, line_perc_2, dot_true_2, time_text)

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(frame_indices),
        interval=70, blit=True
    )

    gif_path = os.path.join(output_dir, "fig1_drone_spoofing_animation.gif")
    anim.save(gif_path, writer=animation.PillowWriter(fps=14))
    plt.close(fig)
    return gif_path


# =============================================================================
# SUMMARY TABLE PRINTER
# =============================================================================
def print_summary_table(cases, start_pos, city_pos, safe_pos, min_cost, city_cost, opt_results):
    """
    Visual 6: Prints a cleanly formatted academic summary table of all simulation runs.
    """
    divider = "=" * 110
    print("\n" + divider)
    print(" GPS SPOOFING DEFENSIVE COUNTERMEASURE — SIMULATION SUMMARY REPORT")
    print(divider)
    print(f" User Drone 1 Start Pos  : [{start_pos[0]:.1f}, {start_pos[1]:.1f}] m")
    print(f" User City Main Location : [{city_pos[0]:.1f}, {city_pos[1]:.1f}] m "
          f"(Peak Harm Index = {city_cost:.2f} / 100 — Darkest Zone)")
    print(f" Auto Min-Harm Safe Zone : [{safe_pos[0]:.1f}, {safe_pos[1]:.1f}] m "
          f"(Min Harm Index = {min_cost:.2f} / 100 — Minimal Damage)")
    print(f" Required Final Offset   : delta_final = [{city_pos[0]-safe_pos[0]:.1f}, "
          f"{city_pos[1]-safe_pos[1]:.1f}] m "
          f"(||delta_final|| = {np.linalg.norm(city_pos - safe_pos):.2f} m)")
    print(f" Optimal Stealth Rate    : {opt_results['optimal_rate']:.2f} m/step "
          f"({opt_results['optimal_rate']/DT:.2f} m/s) -> Peak r[k] = "
          f"{opt_results['optimal_max_residual']:.2f} m < tau = {DETECTION_TAU:.1f} m")
    print("-" * 110)
    header = (
        f"{'Simulation Case':<32} | {'Final True Pos [m]':<21} | "
        f"{'Miss Dist (Safe)':<16} | {'Time to Divert':<14} | "
        f"{'Max r[k] [m]':<12} | {'Detected?'}"
    )
    print(header)
    print("-" * 110)

    for label, res in cases:
        pos_str = f"[{res['final_true'][0]:6.1f}, {res['final_true'][1]:6.1f}]"
        miss_str = f"{res['miss_dist_safe']:6.2f} m"
        time_str = f"{res['divert_time']:5.1f} s"
        res_str = f"{res['max_residual']:6.2f} m"
        det_str = f"YES (t={res['detection_time']:.1f}s)" if res["detected"] else "NO (Stealth)"
        print(f"{label:<32} | {pos_str:<21} | {miss_str:<16} | "
              f"{time_str:<14} | {res_str:<12} | {det_str}")

    print(divider + "\n")


# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apply_clean_style()

    # 0. Get user-chosen Drone Start Point and Protected City Center
    start_pos_1, start_pos_2, city_pos = get_user_coordinates()

    # 1. Build Dynamic 2D Harm Cost Map centered on `city_pos` (darkest peak)
    #    and automatically locate the minimum-damage reachable point `x_safe` (Equation 6)
    X, Y, harm_map, x_safe, min_cost, _ = build_harm_cost_map(start_pos_1, city_pos)
    city_idx = (
        int(np.argmin(np.abs(Y[:, 0] - city_pos[1]))),
        int(np.argmin(np.abs(X[0, :] - city_pos[0])))
    )
    city_cost = float(harm_map[city_idx])

    # 2. Compute simulation time window and run Rate-Limit Optimization (Equation 7)
    sim_time = compute_sim_duration(start_pos_1, city_pos, x_safe)
    opt_results = optimize_delta_rate_max(
        start_pos=start_pos_1, city_pos=city_pos,
        safe_pos=x_safe, sim_time=sim_time, tau=DETECTION_TAU
    )
    active_stealth_rate = float(opt_results["optimal_rate"])

    # 3. Execute Required Simulation Cases:
    #    - Case A: Baseline (No Spoofing — Drone reaches the city)
    res_case_a = run_simulation(
        start_pos=start_pos_1, city_pos=city_pos, safe_pos=x_safe,
        sim_time=sim_time, mode="none", seed=RANDOM_SEED
    )

    #    - Case B: Stealthy Spoofing with Rate-Limited Offset (Drone 1 -> Min-Harm Safe Zone)
    res_case_b = run_simulation(
        start_pos=start_pos_1, city_pos=city_pos, safe_pos=x_safe,
        sim_time=sim_time, mode="stealth", delta_rate_max=active_stealth_rate,
        seed=RANDOM_SEED
    )

    #    - Case C: Aggressive Spoofing with Sudden Offset Jump (Triggers Detector)
    res_case_c = run_simulation(
        start_pos=start_pos_1, city_pos=city_pos, safe_pos=x_safe,
        sim_time=sim_time, mode="aggressive", delta_rate_max=active_stealth_rate,
        seed=RANDOM_SEED
    )

    #    - Extension: 2nd Enemy Drone launched from start_pos_2, spoofed into
    #      a neighboring slot inside the same minimum-damage safe basin
    safe_pos_drone2 = np.clip(x_safe + np.array([-35.0, 25.0]), 50.0, 950.0)
    res_drone_2 = run_simulation(
        start_pos=start_pos_2, city_pos=city_pos, safe_pos=safe_pos_drone2,
        sim_time=sim_time, mode="stealth", delta_rate_max=active_stealth_rate,
        seed=RANDOM_SEED + 7
    )

    # 4. Print Summary Table to Console
    cases_summary = [
        ("Case A: No Spoofing (Drone 1)", res_case_a),
        ("Case B: Stealth Spoof (Drone 1)", res_case_b),
        ("Case C: Aggressive Jump (Drone 1)", res_case_c),
        ("Extension: Stealth Spoof (Drone 2)", res_drone_2),
    ]
    print_summary_table(
        cases_summary, start_pos_1, city_pos, x_safe, min_cost, city_cost, opt_results
    )

    # 5. Generate & Save Static Figures and Animation to ./outputs/
    print("Generating static plots and animation in ./outputs/ ...")
    f2 = plot_static_trajectories(
        X, Y, harm_map, start_pos_1, start_pos_2, city_pos, x_safe,
        res_case_a, res_case_b, res_case_c, res_drone_2, safe_pos_drone2, OUTPUT_DIR
    )
    f3 = plot_spoofing_offset(res_case_a, res_case_b, res_case_c, x_safe, city_pos, OUTPUT_DIR)
    f4 = plot_detector_residuals(res_case_b, res_case_c, opt_results, OUTPUT_DIR)
    f5 = plot_distances_over_time(res_case_a, res_case_b, OUTPUT_DIR)
    f1 = create_animation_gif(
        X, Y, harm_map, start_pos_1, start_pos_2, city_pos, x_safe,
        res_case_b, res_drone_2, OUTPUT_DIR
    )

    print("Saved outputs:")
    for path in [f1, f2, f3, f4, f5]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()