"""
SNAIL PID Controller
====================
Scrubber with Negative-pressure Adhesion and Integrated Locomotion

Control Architecture
--------------------
Outer loop  — Pose control     : (x, y, ψ) error → reference velocities (v*, ω*)
                                   Heading Pose PID  : ψ error   → ω*
                                   Distance Pose PID : dist error → v*

Inner loop  — Velocity control : velocity error → accel cmd → DDR → tilt (θ₁, θ₂)
                                   Velocity PID : (v* - v_meas), (ω* - ω_meas) → accel

Feedforward — VBA adhesion     : φ (IMU) → Eq. 5 lookup → F_normal command
                                   OPEN LOOP — no F_adh sensor, no feedback.
                                   Inclination φ drives F_normal directly via
                                   tiltable shaft + ERM motor.

Reference: Lee et al., "SNAIL: Scrubber with Negative-pressure Adhesion
and Integrated Locomotion for Multi-orientation Surface Cleaning", ICRA 2027.

Authors    : Shawndy Michael Lee et al., SUTD
Controller : [Your name]
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional
from waypointGen import multipoint_waypoints, circle_waypoints, figure8_waypoints, boustrophedon_waypoints

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SNAIL")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class LocomotionMode(Enum):
    """Locomotion modes derived from Table I of the paper."""
    FORWARD  = auto()
    BACKWARD = auto()
    CW       = auto()
    CCW      = auto()
    STOP     = auto()


class SurfaceOrientation(Enum):
    """Surface inclination regime."""
    HORIZONTAL = auto()   # φ ≈ 0°
    INCLINED   = auto()   # 0° < φ < 90°
    VERTICAL   = auto()   # φ ≈ 90°


# ---------------------------------------------------------------------------
# Robot parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SNAILParams:
    """
    Robot physical and system parameters.
    Values from Table II of the paper where available.
    """
    # --- Robot geometry ---
    robot_mass:     float = 1.500    # kg
    hose_mass:      float = 0.500    # kg
    wheel_base:     float = 0.205    # m
    pad_radius:     float = 0.065    # m

    # --- VBA mechanism (Table II) ---
    vba_disk_radius:    float = 0.050
    vba_disk_thickness: float = 0.0005
    vba_omega:          float = 2 * math.pi * 190   # rad/s — 190 Hz ERM
    vba_amplitude:      float = 0.0003              # m
    vba_gap_nominal:    float = 0.0002              # m
    n_vba_disks:        int   = 2

    # --- Fluid / turbine (Table II) ---
    nozzle_diameter: float = 0.0015
    turbine_radius:  float = 0.026
    pump_pressure:   float = 12e6
    mass_flow_rate:  float = 0.103
    fluid_viscosity: float = 7.97e-4  # Ns/m²

    # --- Surface interaction ---
    foam_spring_const: float = 1500.0
    friction_coeff:    float = 0.45

    # --- Operational limits (Section III-D) ---
    max_pad_angle:  float = math.radians(1.0)
    max_tilt_angle: float = math.radians(15.0)

    # --- Velocity limits ---
    max_linear_vel:  float = 0.10   # m/s
    max_angular_vel: float = 1.50   # rad/s

    # --- VBA safety margin ---
    vba_safety_margin: float = 1.5  # F_normal target = 1.5 × F_min (Eq. 5)

    # --- Output smoothing (tune these on hardware) ---
    # Low-pass filter alpha: 0.0 = max smooth (most lag), 1.0 = no filtering
    lpf_alpha:        float = 0.20
    # Rate limiter: max pad tilt change rate (deg/s). Lower = smoother, slower.
    tilt_rate_limit:  float = 25.0
    # Heading deadband: ψ errors below this are treated as zero (deg).
    heading_deadband: float = 3.0

    # --- Physical constants ---
    gravity:   float = 9.81
    rho_fluid: float = 1000.0


# ---------------------------------------------------------------------------
# PID core
# ---------------------------------------------------------------------------
@dataclass
class PIDState:
    integral:   float = 0.0
    prev_error: float = 0.0


class PIDController:
    """
    Standard PID with:
      - Anti-windup via back-calculation
      - Derivative low-pass filter (time constant τ_d)
      - Output saturation
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -float("inf"),
        output_max: float =  float("inf"),
        integral_limit: float = float("inf"),
        derivative_tau: float = 0.0,
        name: str = "PID",
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        self.derivative_tau = derivative_tau
        self.name = name
        self._state = PIDState()
        self._d_filtered: float = 0.0

    def reset(self) -> None:
        self._state = PIDState()
        self._d_filtered = 0.0
        logger.debug(f"[{self.name}] Reset.")

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            logger.warning(f"[{self.name}] dt={dt:.6f} ≤ 0, skipping.")
            return 0.0

        error = setpoint - measurement

        p_term = self.kp * error

        self._state.integral += error * dt
        self._state.integral = float(np.clip(
            self._state.integral, -self.integral_limit, self.integral_limit
        ))
        i_term = self.ki * self._state.integral

        d_raw = (error - self._state.prev_error) / dt
        if self.derivative_tau > 0.0:
            alpha = dt / (self.derivative_tau + dt)
            self._d_filtered = alpha * d_raw + (1.0 - alpha) * self._d_filtered
        else:
            self._d_filtered = d_raw
        d_term = self.kd * self._d_filtered

        raw_output = p_term + i_term + d_term
        output = float(np.clip(raw_output, self.output_min, self.output_max))

        if output != raw_output:
            self._state.integral -= error * dt   # back-calculation anti-windup

        self._state.prev_error = error
        logger.debug(
            f"[{self.name}] e={error:+.4f}  P={p_term:+.4f}  "
            f"I={i_term:+.4f}  D={d_term:+.4f}  u={output:+.4f}"
        )
        return output



# ---------------------------------------------------------------------------
# Output smoothing utilities
# ---------------------------------------------------------------------------
class LowPassFilter:
    """
    First-order low-pass filter: y[k] = alpha * x[k] + (1 - alpha) * y[k-1]

    Tune via SNAILParams.lpf_alpha:
      - Lower alpha (e.g. 0.10) -> smoother output, more tracking lag
      - Higher alpha (e.g. 0.40) -> faster response, more noise passes through

    Applied to: v*, omega*, theta1, theta2
    """
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._y: Optional[float] = None

    def reset(self) -> None:
        self._y = None

    def __call__(self, x: float) -> float:
        if self._y is None:
            self._y = x
        self._y = self.alpha * x + (1.0 - self.alpha) * self._y
        return self._y


class RateLimiter:
    """
    Clamps the rate of change of a signal to +/- max_rate per second.

    Tune via SNAILParams.tilt_rate_limit (deg/s):
      - Lower  (e.g. 15 deg/s) -> smooth tilt transitions, slower turns
      - Higher (e.g. 45 deg/s) -> faster response, more abrupt changes

    Applied to: theta1, theta2 (after LPF)
    """
    def __init__(self, max_rate_rad_per_s: float) -> None:
        self.max_rate = max_rate_rad_per_s
        self._prev: Optional[float] = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, x: float, dt: float) -> float:
        if self._prev is None:
            self._prev = x
        delta = float(np.clip(
            x - self._prev,
            -self.max_rate * dt,
             self.max_rate * dt,
        ))
        self._prev += delta
        return self._prev


# ---------------------------------------------------------------------------
# VBA Adhesion Force Model  (Eq. 2 & 4)
# ---------------------------------------------------------------------------
class VBAAdhesionModel:
    """
    Physics model for VBA adhesion force (Eq. 2 / 4).
    Used by the feedforward block to compute F_normal requirements.
    """

    def __init__(self, params: SNAILParams) -> None:
        self.p = params

    def compute_adhesion_force(self, gap: float, wet: bool = True) -> float:
        """Eq. 2 / 4: F_adh = (3π η ω² A² R⁴) / (4 h³)"""
        eta   = self.p.fluid_viscosity if wet else 1.81e-5
        h     = max(gap, 1e-6)
        f_per = (3.0 * math.pi * eta * self.p.vba_omega**2
                 * self.p.vba_amplitude**2 * self.p.vba_disk_radius**4) / (4.0 * h**3)
        return self.p.n_vba_disks * f_per

    def minimum_required_force(self, inclination_rad: float) -> float:
        """
        Eq. 5 — minimum adhesion to prevent gravitational sliding:
            F_adh ≥ (m_r + m_h) g sin(φ)

        Note: necessary but not sufficient — the coupled traction constraint
            F_adh ≥ F_fric / μ
        must also hold; the binding condition depends on φ.
        """
        m = self.p.robot_mass + self.p.hose_mass
        return m * self.p.gravity * math.sin(inclination_rad)

    def surface_orientation(self, inclination_rad: float) -> SurfaceOrientation:
        deg = math.degrees(inclination_rad)
        if deg < 5.0:   return SurfaceOrientation.HORIZONTAL
        if deg < 85.0:  return SurfaceOrientation.INCLINED
        return SurfaceOrientation.VERTICAL


# ---------------------------------------------------------------------------
# VBA Feedforward Block  (open loop — no F_adh sensor)
# ---------------------------------------------------------------------------
class VBAFeedforward:
    """
    Open-loop feedforward for VBA adhesion.

    Since F_adh cannot be measured directly, there is no feedback loop.
    Instead, the surface inclination φ (from IMU) is used to compute the
    required normal force via Eq. 5, scaled by a safety margin, and
    commanded directly to the tiltable shaft + ERM motor.

    Architecture:
        φ (IMU)  →  Eq. 5 lookup  →  F_normal command  →  ERM + tiltable shaft
        [no Σ junction, no PID, no feedback arrow]
    """

    def __init__(self, params: Optional[SNAILParams] = None) -> None:
        self.params    = params or SNAILParams()
        self.vba_model = VBAAdhesionModel(self.params)

    def compute(self, inclination_rad: float) -> "AdhesionFeedforward":
        """
        Compute the required F_normal from inclination alone.

        Args:
            inclination_rad: Surface inclination φ measured by IMU (rad).

        Returns:
            AdhesionFeedforward with F_normal command and orientation.
        """
        f_min        = self.vba_model.minimum_required_force(inclination_rad)
        f_normal_cmd = f_min * self.params.vba_safety_margin
        orientation  = self.vba_model.surface_orientation(inclination_rad)

        logger.debug(
            f"[VBA FF] φ={math.degrees(inclination_rad):.1f}°  "
            f"F_min={f_min:.2f}N  F_normal_cmd={f_normal_cmd:.2f}N  "
            f"[{orientation.name}]"
        )

        return AdhesionFeedforward(
            f_normal_cmd=f_normal_cmd,
            f_adh_min=f_min,
            orientation=orientation,
        )


# ---------------------------------------------------------------------------
# State and command dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RobotState:
    """Full observable state passed to the controller each cycle."""
    x:      float = 0.0   # m       — position
    y:      float = 0.0   # m
    psi:    float = 0.0   # rad     — heading
    phi:    float = 0.0   # rad     — surface inclination (from IMU)
    v_meas: float = 0.0   # m/s     — measured linear velocity
    w_meas: float = 0.0   # rad/s   — measured angular velocity


@dataclass
class LocomotionCommand:
    """Output of the cascaded pose + velocity loops → pad actuators."""
    theta1:      float          # pad 1 tilt angle (rad)
    theta2:      float          # pad 2 tilt angle (rad)
    v_ref:       float          # commanded linear velocity (m/s)
    omega_ref:   float          # commanded angular velocity (rad/s)
    mode:        LocomotionMode
    orientation: SurfaceOrientation


@dataclass
class AdhesionFeedforward:
    """
    Output of the VBA open-loop feedforward block → ERM + tiltable shaft.
    No feedback, no error signal — purely derived from φ via Eq. 5.
    """
    f_normal_cmd: float          # F_normal to apply (N)
    f_adh_min:    float          # minimum required per Eq. 5 (N)
    orientation:  SurfaceOrientation


# ---------------------------------------------------------------------------
# Locomotion model  (Eq. 9a, 10a, 15)
# ---------------------------------------------------------------------------
class LocomotionModel:
    """DDR decomposition + Eq. 15 tilt mapping + Table I mode classification."""

    def __init__(self, params: SNAILParams) -> None:
        self.p = params

    def velocity_to_tilt(self, velocity: float, omega_pad: float) -> float:
        ratio = velocity / max(omega_pad * self.p.pad_radius, 1e-9)
        return float(np.arcsin(np.clip(ratio, -1.0, 1.0)))

    def ddrive_to_pad_commands(
        self, v_linear: float, v_angular: float, omega_pad: float
    ) -> Tuple[float, float]:
        L = self.p.wheel_base
        v_right =  v_linear + (v_angular * L / 2.0)
        v_left  =  v_linear - (v_angular * L / 2.0)
        theta1 = self.velocity_to_tilt(-v_left,  omega_pad)
        theta2 = self.velocity_to_tilt( v_right, omega_pad)
        theta1 = float(np.clip(theta1, -self.p.max_tilt_angle, self.p.max_tilt_angle))
        theta2 = float(np.clip(theta2, -self.p.max_tilt_angle, self.p.max_tilt_angle))
        return theta1, theta2

    def locomotion_mode(self, theta1: float, theta2: float) -> LocomotionMode:
        t1p, t1n = theta1 > 1e-3, theta1 < -1e-3
        t2p, t2n = theta2 > 1e-3, theta2 < -1e-3
        if   t1n and t2p: return LocomotionMode.FORWARD
        elif t1p and t2n: return LocomotionMode.BACKWARD
        elif t1p and t2p: return LocomotionMode.CW
        elif t1n and t2n: return LocomotionMode.CCW
        else:             return LocomotionMode.STOP


# ---------------------------------------------------------------------------
# Cascaded Locomotion Controller  (Outer Pose + Inner Velocity loops)
# ---------------------------------------------------------------------------
class SNAILLocomotionController:
    """
    Two-layer cascaded PID for locomotion.

    Outer loop — Pose PIDs:
        Heading Pose PID : ψ error   → ω* (reference yaw rate)
        Distance Pose PID: dist error → v* (reference linear velocity)

    Inner loop — Velocity PID:
        Velocity PID : (v* - v_meas), (ω* - ω_meas) → acceleration commands
                       → DDR decomposition → tilt mapping (Eq. 15) → (θ₁, θ₂)
    """

    DEFAULT_GAINS = {
        # Outer loop — Pose PIDs
        "heading_pose":  dict(kp=2.50, ki=0.05, kd=0.30,
                              output_min=-1.5,  output_max=1.5,
                              integral_limit=0.5, derivative_tau=0.05),
        "distance_pose": dict(kp=1.20, ki=0.10, kd=0.05,
                              output_min=-0.10, output_max=0.10,
                              integral_limit=0.2, derivative_tau=0.02),
        # Inner loop — Velocity PIDs
        "velocity_lin":  dict(kp=3.00, ki=0.20, kd=0.10,
                              output_min=-0.20, output_max=0.20,
                              integral_limit=0.3, derivative_tau=0.01),
        "velocity_ang":  dict(kp=2.50, ki=0.15, kd=0.08,
                              output_min=-2.0,  output_max=2.0,
                              integral_limit=0.5, derivative_tau=0.01),
    }

    def __init__(
        self,
        params:      Optional[SNAILParams] = None,
        gains:       Optional[dict] = None,
        omega_pad:   float = 50.0,
        goal_radius: float = 0.02,
    ) -> None:
        self.params      = params or SNAILParams()
        self.omega_pad   = omega_pad
        self.goal_radius = goal_radius

        g = gains or self.DEFAULT_GAINS
        # Outer loop — Pose PIDs
        self.pid_heading_pose  = PIDController(**g["heading_pose"],  name="Heading_Pose_PID")
        self.pid_distance_pose = PIDController(**g["distance_pose"], name="Distance_Pose_PID")
        # Inner loop — Velocity PIDs
        self.pid_velocity_lin  = PIDController(**g["velocity_lin"],  name="Velocity_PID_Lin")
        self.pid_velocity_ang  = PIDController(**g["velocity_ang"],  name="Velocity_PID_Ang")

        self.loco_model = LocomotionModel(self.params)
        self.vba_model  = VBAAdhesionModel(self.params)

        # Output smoothing — tune via SNAILParams
        a = self.params.lpf_alpha
        self._lpf_vref  = LowPassFilter(a)
        self._lpf_wref  = LowPassFilter(a)
        self._lpf_t1    = LowPassFilter(a)
        self._lpf_t2    = LowPassFilter(a)
        rl = math.radians(self.params.tilt_rate_limit)
        self._rl_t1     = RateLimiter(rl)
        self._rl_t2     = RateLimiter(rl)

        self._goal_x:    float = 0.0
        self._goal_y:    float = 0.0
        self._v_ref:     float = 0.0
        self._omega_ref: float = 0.0
        self._last_t_outer: float = time.monotonic()
        self._last_t_inner: float = time.monotonic()

    def set_goal(self, x: float, y: float) -> None:
        self._goal_x = x
        self._goal_y = y
        self.pid_heading_pose.reset()
        self.pid_distance_pose.reset()
        self.pid_velocity_lin.reset()
        self.pid_velocity_ang.reset()
        for f in [self._lpf_vref, self._lpf_wref,
                  self._lpf_t1, self._lpf_t2,
                  self._rl_t1, self._rl_t2]:
            f.reset()
        logger.info(f"New goal -> ({x:.3f}, {y:.3f}) m")

    def goal_reached(self, state: RobotState) -> bool:
        return math.hypot(self._goal_x - state.x, self._goal_y - state.y) < self.goal_radius

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def step_outer(self, state: RobotState) -> Tuple[float, float]:
        """
        Outer pose loop: pose error → reference velocities (v*, ω*).
        Typical rate: ~20 Hz.
        """
        now = time.monotonic()
        dt  = max(now - self._last_t_outer, 1e-4)
        self._last_t_outer = now

        dx, dy       = self._goal_x - state.x, self._goal_y - state.y
        dist         = math.hypot(dx, dy)
        heading_err  = self._wrap_angle(math.atan2(dy, dx) - state.psi)
        # Heading deadband — kills jitter on near-straight segments
        # Tune: SNAILParams.heading_deadband (deg). Larger = less correction noise,
        #       but robot may drift slightly off the straight-line path.
        db = math.radians(self.params.heading_deadband)
        if abs(heading_err) < db:
            heading_err = 0.0
        alignment    = max(0.0, math.cos(heading_err))

        omega_raw       = self.pid_heading_pose.compute(0.0, -heading_err, dt)
        v_raw           = self.pid_distance_pose.compute(dist * alignment, 0.0, dt)
        # Low-pass filter on reference outputs
        # Tune: SNAILParams.lpf_alpha
        self._omega_ref = self._lpf_wref(omega_raw)
        self._v_ref     = self._lpf_vref(v_raw)

        logger.debug(
            f"[Outer] dist={dist:.3f}m  ψ_err={math.degrees(heading_err):+.1f}°  "
            f"v*={self._v_ref:+.3f}  ω*={self._omega_ref:+.3f}"
        )
        return self._v_ref, self._omega_ref

    def step_inner(self, state: RobotState) -> LocomotionCommand:
        """
        Inner velocity loop: velocity error → pad tilt commands.
        Typical rate: ~50–100 Hz.
        """
        now = time.monotonic()
        dt  = max(now - self._last_t_inner, 1e-4)
        self._last_t_inner = now

        v_cmd = self.pid_velocity_lin.compute(self._v_ref,     state.v_meas, dt)
        w_cmd = self.pid_velocity_ang.compute(self._omega_ref, state.w_meas, dt)

        v_cmd = float(np.clip(v_cmd, -self.params.max_linear_vel,  self.params.max_linear_vel))
        w_cmd = float(np.clip(w_cmd, -self.params.max_angular_vel, self.params.max_angular_vel))

        theta1_raw, theta2_raw = self.loco_model.ddrive_to_pad_commands(v_cmd, w_cmd, self.omega_pad)
        # Low-pass filter then rate-limit the tilt angles before sending to actuators.
        # Tune: SNAILParams.lpf_alpha and SNAILParams.tilt_rate_limit
        theta1 = self._rl_t1(self._lpf_t1(theta1_raw), dt)
        theta2 = self._rl_t2(self._lpf_t2(theta2_raw), dt)
        mode        = self.loco_model.locomotion_mode(theta1, theta2)
        orientation = self.vba_model.surface_orientation(state.phi)

        logger.info(
            f"[Inner] v*={self._v_ref:+.3f} v={state.v_meas:+.3f}  "
            f"ω*={self._omega_ref:+.3f} ω={state.w_meas:+.3f}  "
            f"θ=({math.degrees(theta1):+.1f}°,{math.degrees(theta2):+.1f}°)  "
            f"mode={mode.name}"
        )

        return LocomotionCommand(
            theta1=theta1, theta2=theta2,
            v_ref=v_cmd, omega_ref=w_cmd,
            mode=mode, orientation=orientation,
        )

    def step(self, state: RobotState) -> LocomotionCommand:
        """Run outer then inner at equal rate (simulation / simple deployment)."""
        self.step_outer(state)
        return self.step_inner(state)


# ---------------------------------------------------------------------------
# Top-level coordinator
# ---------------------------------------------------------------------------
class SNAILController:
    """
    Top-level coordinator for SNAIL.

    Owns:
      - SNAILLocomotionController  (outer Pose PID + inner Velocity PID cascade)
      - VBAFeedforward             (open-loop feedforward — no sensor, no PID)

    Typical real-time usage
    -----------------------
    controller = SNAILController()
    controller.set_goal(1.0, 0.5)

    # Each control tick (e.g. 50 Hz):
    loco_cmd = controller.step(state)
    set_pad_tilt(loco_cmd.theta1, loco_cmd.theta2)

    # VBA feedforward — recompute whenever φ changes (e.g. 20 Hz with IMU):
    adh_cmd = controller.compute_adhesion(state.phi)
    set_fnormal(adh_cmd.f_normal_cmd)
    """

    def __init__(
        self,
        params:      Optional[SNAILParams] = None,
        loco_gains:  Optional[dict] = None,
        omega_pad:   float = 50.0,
        goal_radius: float = 0.02,
    ) -> None:
        p = params or SNAILParams()
        self.locomotion = SNAILLocomotionController(
            params=p, gains=loco_gains,
            omega_pad=omega_pad, goal_radius=goal_radius,
        )
        self.vba_feedforward = VBAFeedforward(params=p)
        logger.info("SNAILController initialised (VBA: open-loop feedforward).")

    def set_goal(self, x: float, y: float) -> None:
        self.locomotion.set_goal(x, y)

    def goal_reached(self, state: RobotState) -> bool:
        return self.locomotion.goal_reached(state)

    def step_outer(self, state: RobotState) -> Tuple[float, float]:
        return self.locomotion.step_outer(state)

    def step_inner(self, state: RobotState) -> LocomotionCommand:
        return self.locomotion.step_inner(state)

    def compute_adhesion(self, inclination_rad: float) -> AdhesionFeedforward:
        """
        Compute F_normal command from inclination (open-loop feedforward).
        Call whenever φ is updated from the IMU — no fixed rate required.
        """
        return self.vba_feedforward.compute(inclination_rad)

    def step(self, state: RobotState) -> Tuple[LocomotionCommand, AdhesionFeedforward]:
        """
        Run full control cycle (equal-rate, for simulation).
        Returns (LocomotionCommand, AdhesionFeedforward).
        """
        loco = self.locomotion.step(state)
        adh  = self.vba_feedforward.compute(state.phi)
        return loco, adh


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------
def simulate(
    waypoints: list[Tuple[float, float]],
    inclination_deg: float = 0.0,
    params: Optional[SNAILParams] = None,
    dt_outer: float = 0.05,
    dt_inner: float = 0.01,
    dt_imu:   float = 0.05,
    max_steps: int = 10000,
) -> dict:
    """
    Multi-rate simulation of SNAIL navigating a waypoint sequence.
    Logs both raw (pre-filter) and clean (post-filter) signals for comparison.

    Args:
        waypoints:       List of (x, y) targets in metres.
        inclination_deg: Surface tilt angle in degrees.
        params:          SNAILParams — pass custom params to change smoothing.
        dt_outer/inner/imu: Loop periods in seconds.
        max_steps:       Safety cap per waypoint.

    Returns:
        Dict with raw and clean signal histories.
    """
    p          = params or SNAILParams()
    controller = SNAILController(params=p)
    phi        = math.radians(inclination_deg)
    state      = RobotState(x=0.0, y=0.0, psi=0.0, phi=phi)

    history: dict = {
        "x": [], "y": [], "psi": [],
        # Raw (pre-filter) signals
        "v_ref_raw": [], "omega_ref_raw": [],
        "theta1_raw": [], "theta2_raw": [],
        # Clean (post-filter) signals
        "v_ref": [], "omega_ref": [],
        "theta1": [], "theta2": [],
        # Measured
        "v_meas": [], "w_meas": [],
        # VBA
        "f_normal_cmd": [], "f_adh_min": [],
        "mode": [], "t": [],
    }

    t            = 0.0
    t_last_outer = -dt_outer
    t_last_inner = -dt_inner
    t_last_imu   = -dt_imu
    adh_cmd      = controller.compute_adhesion(phi)
    loco_cmd     = None

    for wp in waypoints:
        controller.set_goal(*wp)
        logger.info(f"=== Navigating to waypoint {wp} ===")

        for _ in range(max_steps):

            # ── VBA feedforward ───────────────────────────────────────
            if t - t_last_imu >= dt_imu:
                adh_cmd    = controller.compute_adhesion(state.phi)
                t_last_imu = t

            # ── Outer pose loop ───────────────────────────────────────
            if t - t_last_outer >= dt_outer:
                controller.step_outer(state)
                t_last_outer = t

            # ── Inner velocity loop ───────────────────────────────────
            if t - t_last_inner >= dt_inner:
                loco_cmd     = controller.step_inner(state)
                t_last_inner = t

            # ── Plant integration ─────────────────────────────────────
            v_cmd = loco_cmd.v_ref     if loco_cmd else state.v_meas
            w_cmd = loco_cmd.omega_ref if loco_cmd else state.w_meas

            tau = 0.05
            state.v_meas += (v_cmd - state.v_meas) * (dt_inner / tau)
            state.w_meas += (w_cmd - state.w_meas) * (dt_inner / tau)
            state.x      += state.v_meas * math.cos(state.psi) * dt_inner
            state.y      += state.v_meas * math.sin(state.psi) * dt_inner
            state.psi    += state.w_meas * dt_inner
            state.psi     = (state.psi + math.pi) % (2.0 * math.pi) - math.pi

            # ── Log both raw and clean signals ────────────────────────
            loco = controller.locomotion
            th1  = loco_cmd.theta1     if loco_cmd else 0.0
            th2  = loco_cmd.theta2     if loco_cmd else 0.0
            md   = loco_cmd.mode.name  if loco_cmd else "HOLD"

            history["x"].append(state.x)
            history["y"].append(state.y)
            history["psi"].append(math.degrees(state.psi))
            # Clean (post-filter) — what is actually sent to hardware
            history["v_ref"].append(loco._v_ref)
            history["omega_ref"].append(loco._omega_ref)
            history["theta1"].append(math.degrees(th1))
            history["theta2"].append(math.degrees(th2))
            # Raw (pre-filter) — what the PID computed before smoothing
            history["v_ref_raw"].append(loco.pid_distance_pose._state.prev_error * -loco.pid_distance_pose.kp if loco_cmd else 0.0)
            history["omega_ref_raw"].append(loco.pid_heading_pose._state.prev_error * -loco.pid_heading_pose.kp if loco_cmd else 0.0)
            history["theta1_raw"].append(math.degrees(loco._lpf_t1._y or 0.0))  # post-LPF pre-RL
            history["theta2_raw"].append(math.degrees(loco._lpf_t2._y or 0.0))
            history["v_meas"].append(state.v_meas)
            history["w_meas"].append(state.w_meas)
            history["f_normal_cmd"].append(adh_cmd.f_normal_cmd)
            history["f_adh_min"].append(adh_cmd.f_adh_min)
            history["mode"].append(md)
            history["t"].append(t)

            t += dt_inner

            if controller.goal_reached(state):
                logger.info(f"Waypoint {wp} reached in {t:.2f}s")
                break

    return history


# ---------------------------------------------------------------------------
# Monte-Carlo trial runner  (Parameter perturbation)
# ---------------------------------------------------------------------------
def simulate_trials(
    waypoints:       list[Tuple[float, float]],
    inclination_deg: float = 0.0,
    n_runs:          int   = 8,
    seed:            int   = 42,
    dt_outer:        float = 0.05,
    dt_inner:        float = 0.01,
    max_steps:       int   = 10000,
) -> dict:
    """
    Run ``n_runs`` independent trials of the same waypoint sequence, each
    with a slightly different parameter set drawn from physically-motivated
    distributions.  This produces realistic trial-to-trial spread for the
    ±1 SD error bands in positionvstime.py.

    Perturbation model (all sampled independently per trial):
        psi_0         ~ N(0, 8°)        initial heading offset (encoder zeroing error)
        lpf_alpha     ~ N(0.20, 0.08)   filter bandwidth variation (component tolerance)
        heading_db    ~ N(3.0°, 1.5°)   deadband jitter (tuning uncertainty)
        friction_coef ~ N(0.45, 0.10)   surface friction variation (wetness, fouling)
        v_scale       ~ N(1.0, 0.08)    drive velocity scale (motor voltage variation)

    Args:
        waypoints:       Waypoint list passed to simulate().
        inclination_deg: Surface tilt (°).
        n_runs:          Number of independent trials.
        seed:            RNG seed for reproducibility.
        dt_outer/inner:  Loop periods (s).
        max_steps:       Safety cap per waypoint.

    Returns:
        Dict with keys ``t``, ``x``, ``y`` — each a (n_runs × steps) ndarray,
        trimmed to the shortest completed run so all arrays are the same length.
        Also includes ``x_mean``, ``x_std``, ``y_mean``, ``y_std``, ``t_common``.
    """
    rng = np.random.default_rng(seed)
    runs_t, runs_x, runs_y = [], [], []

    for run_i in range(n_runs):
        # ── Sample perturbed parameters ───────────────────────────────────
        psi_0      = rng.normal(0.0,  math.radians(8.0))   # rad
        alpha      = float(np.clip(rng.normal(0.20, 0.08),  0.05, 0.60))
        deadband   = float(np.clip(rng.normal(3.0,  1.5),   0.5,  8.0))
        friction   = float(np.clip(rng.normal(0.45, 0.10),  0.20, 0.80))
        v_scale    = float(np.clip(rng.normal(1.0,  0.08),  0.75, 1.25))

        p = SNAILParams(
            lpf_alpha=alpha,
            heading_deadband=deadband,
            friction_coeff=friction,
        )

        # ── Build a velocity-scaled controller by monkey-patching plant ──
        # v_scale is applied in the plant integrator below via a wrapper
        phi   = math.radians(inclination_deg)
        ctrl  = SNAILController(params=p)
        state = RobotState(x=0.0, y=0.0, psi=psi_0, phi=phi)

        t_log, x_log, y_log = [], [], []
        t            = 0.0
        t_last_outer = -dt_outer
        t_last_inner = -dt_inner
        t_last_imu   = -dt_outer
        adh_cmd      = ctrl.compute_adhesion(phi)
        loco_cmd     = None

        for wp in waypoints:
            ctrl.set_goal(*wp)
            for _ in range(max_steps):
                if t - t_last_imu >= dt_outer:
                    adh_cmd    = ctrl.compute_adhesion(state.phi)
                    t_last_imu = t
                if t - t_last_outer >= dt_outer:
                    ctrl.step_outer(state)
                    t_last_outer = t
                if t - t_last_inner >= dt_inner:
                    loco_cmd     = ctrl.step_inner(state)
                    t_last_inner = t

                v_cmd = (loco_cmd.v_ref     * v_scale) if loco_cmd else state.v_meas
                w_cmd =  loco_cmd.omega_ref             if loco_cmd else state.w_meas

                tau = 0.05
                state.v_meas += (v_cmd - state.v_meas) * (dt_inner / tau)
                state.w_meas += (w_cmd - state.w_meas) * (dt_inner / tau)

                # Per-step process noise — models wheel slip, surface irregularity,
                # actuator deadband.  Scales with inclination (rougher at high phi).
                noise_scale = 0.003 + math.radians(inclination_deg) / math.pi * 0.008
                state.x   += (state.v_meas * math.cos(state.psi)
                               + rng.normal(0, noise_scale)) * dt_inner
                state.y   += (state.v_meas * math.sin(state.psi)
                               + rng.normal(0, noise_scale)) * dt_inner
                state.psi += (state.w_meas
                               + rng.normal(0, noise_scale * 2)) * dt_inner
                state.psi  = (state.psi + math.pi) % (2.0 * math.pi) - math.pi

                t_log.append(t)
                x_log.append(state.x)
                y_log.append(state.y)
                t += dt_inner

                if ctrl.goal_reached(state):
                    break

        runs_t.append(t_log)
        runs_x.append(x_log)
        runs_y.append(y_log)
        logger.info(f"[Trial {run_i+1}/{n_runs}] θ={inclination_deg}°  "
                    f"psi0={math.degrees(psi_0):+.1f}°  α={alpha:.2f}  "
                    f"db={deadband:.1f}°  μ={friction:.2f}  v×={v_scale:.2f}  "
                    f"steps={len(t_log)}")

    # ── Align to shortest run ──────────────────────────────────────────────
    min_len = min(len(r) for r in runs_x)
    xs = np.array([r[:min_len] for r in runs_x])   # (n_runs, steps)
    ys = np.array([r[:min_len] for r in runs_y])
    t  = np.array(runs_t[0][:min_len])

    return {
        "t":       t,
        "x":       xs,
        "y":       ys,
        "x_mean":  xs.mean(axis=0),
        "x_std":   xs.std(axis=0),
        "y_mean":  ys.mean(axis=0),
        "y_std":   ys.std(axis=0),
    }


def export_trials(
    output_path: str,
    waypoints:   list[Tuple[float, float]],
    thetas:      list[float] = None,
    n_runs:      int         = 8,
    seed:        int         = 42,
    **sim_kwargs,
) -> None:
    """
    Run simulate_trials() for each surface inclination and save results to a
    single .npz archive — ready to be loaded by positionvstime.py.

    File layout (one group per inclination angle):
        traj_<theta>_t        : (steps,)         shared time vector
        traj_<theta>_x        : (n_runs, steps)  per-trial X positions
        traj_<theta>_y        : (n_runs, steps)  per-trial Y positions
        traj_<theta>_x_mean   : (steps,)
        traj_<theta>_x_std    : (steps,)
        traj_<theta>_y_mean   : (steps,)
        traj_<theta>_y_std    : (steps,)

    Usage in positionvstime.py:
        data = np.load("snail_trials.npz")
        t       = data["traj_45_t"]
        x_mean  = data["traj_45_x_mean"]
        x_std   = data["traj_45_x_std"]
        y_mean  = data["traj_45_y_mean"]
        y_std   = data["traj_45_y_std"]

    Args:
        output_path : Filename for the .npz archive (e.g. "snail_trials.npz").
        waypoints   : Waypoint list (same for all inclinations).
        thetas      : List of inclination angles in degrees. Default [0, 45, 90].
        n_runs      : Trials per inclination.
        seed        : Base RNG seed (each theta gets seed + i for independence).
        **sim_kwargs: Forwarded to simulate_trials() (dt_outer, dt_inner, …).
    """
    if thetas is None:
        thetas = [0.0, 45.0, 90.0]

    arrays = {}
    for i, theta in enumerate(thetas):
        key = str(int(theta))
        print(f"[export_trials] θ={theta}°  ({n_runs} runs) …", flush=True)
        result = simulate_trials(
            waypoints, inclination_deg=theta,
            n_runs=n_runs, seed=seed + i, **sim_kwargs,
        )
        arrays[f"traj_{key}_t"]      = result["t"]
        arrays[f"traj_{key}_x"]      = result["x"]
        arrays[f"traj_{key}_y"]      = result["y"]
        arrays[f"traj_{key}_x_mean"] = result["x_mean"]
        arrays[f"traj_{key}_x_std"]  = result["x_std"]
        arrays[f"traj_{key}_y_mean"] = result["y_mean"]
        arrays[f"traj_{key}_y_std"]  = result["y_std"]
        print(f"           done  →  x shape={result['x'].shape}  "
              f"x_std_max={result['x_std'].max():.4f} m  "
              f"y_std_max={result['y_std'].max():.4f} m")

    np.savez(output_path, **arrays)
    print(f"\n[export_trials] Saved → {output_path}")
    print("  Load in positionvstime.py with:")
    print(f'    data = np.load("{output_path}")')
    print('    t, x_mean, x_std = data["traj_45_t"], data["traj_45_x_mean"], data["traj_45_x_std"]')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("=" * 62)
    print("  SNAIL PID Controller — Raw vs Clean Simulation")
    print("=" * 62)

    # --- Example: navigate a circular path on a 45° inclined surface ---
    inclination_deg = 45.0
    waypoints = multipoint_waypoints(
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (0.0, 1.0),
    )
    # waypoints = circle_waypoints(cx=0.0, cy=0.0, radius=0.30, n_points=12)
    # waypoints = figure8_waypoints(cx=0.0, cy=0.0, radius=0.20, n_points=24)
    # waypoints = boustrophedon_waypoints(x0=-0.0, y0=-0.0, width=2.0, height=1.0, lane_width=0.20)
    history   = simulate(waypoints, inclination_deg=inclination_deg, dt_outer=0.05, dt_inner=0.01)

    # ── Run RAW (no smoothing: alpha=1, huge rate limit, zero deadband) ──
    p_raw = SNAILParams(lpf_alpha=1.0, tilt_rate_limit=9999.0, heading_deadband=0.0)
    print("Running RAW simulation...")
    h_raw = simulate(waypoints, inclination_deg=inclination_deg, params=p_raw)

    # ── Run CLEAN (default smoothing params) ──────────────────────────────
    p_cln = SNAILParams()   # uses lpf_alpha=0.20, tilt_rate_limit=25.0, heading_deadband=3.0
    print("Running CLEAN simulation...")
    h_cln = simulate(waypoints, inclination_deg=inclination_deg, params=p_cln)

    print(f"\nRAW   final pos: ({h_raw['x'][-1]:.3f}, {h_raw['y'][-1]:.3f}) m")
    print(f"CLEAN final pos: ({h_cln['x'][-1]:.3f}, {h_cln['y'][-1]:.3f}) m")
    print(f"\nVBA feedforward @ {inclination_deg}°:")
    print(f"  F_adh_min    = {h_cln['f_adh_min'][0]:.3f} N")
    print(f"  F_normal_cmd = {h_cln['f_normal_cmd'][0]:.3f} N  (1.5x safety margin)")

    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          14,
        "axes.titlesize":     14,
        "axes.labelsize":     9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "lines.linewidth":    1.4,
        "axes.linewidth":     0.7,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.35,
        "figure.dpi":         300,
    })

    CR = "#CC3300"; CC = "#0055CC"
    CR2 = "#FF9977"; CC2 = "#66AAFF"

    def fmt(ax, title, xlabel, ylabel):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"SNAIL — Raw vs Clean signals  ({inclination_deg}° incline)", fontweight="bold")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.35)

    # Row 0 — trajectory, v*, ω*, heading
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h_raw["x"], h_raw["y"], color=CR, lw=1.5, label="Raw")
    ax.plot(h_cln["x"], h_cln["y"], color=CC, lw=1.5, ls="--", label="Clean")
    ax.scatter(*zip(*waypoints), color="black", zorder=5, s=30, label="Waypoints")
    fmt(ax, "Trajectory", "x (m)", "y (m)"); ax.legend()

    def twin(gs_pos, tr, yr, tc, yc, title, ylabel):
        ax = fig.add_subplot(gs_pos)
        ax.plot(tr, yr, color=CR, lw=0.8, alpha=0.85, label="Raw")
        ax.plot(tc, yc, color=CC, lw=1.2, ls="--", label="Clean")
        fmt(ax, title, "Time (s)", ylabel); ax.legend()

    twin(gs[0,1], h_raw["t"],h_raw["v_ref"],     h_cln["t"],h_cln["v_ref"],     "v* reference",  "v* (m/s)")
    twin(gs[0,2], h_raw["t"],h_raw["omega_ref"],  h_cln["t"],h_cln["omega_ref"], "ω* reference",  "ω* (rad/s)")
    twin(gs[0,3], h_raw["t"],h_raw["psi"],         h_cln["t"],h_cln["psi"],       "Heading ψ",     "ψ (°)")

    # Row 1 — velocity tracking, yaw tracking, θ₁, θ₂
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(h_raw["t"], h_raw["v_ref"],  color=CR,  lw=0.8, label="v* raw")
    ax.plot(h_raw["t"], h_raw["v_meas"], color=CR2, lw=0.8, ls=":", label="v meas raw")
    ax.plot(h_cln["t"], h_cln["v_ref"],  color=CC,  lw=1.2, ls="--", label="v* clean")
    ax.plot(h_cln["t"], h_cln["v_meas"], color=CC2, lw=1.0, ls="-.", label="v meas clean")
    fmt(ax, "Velocity tracking", "Time (s)", "v (m/s)"); ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(h_raw["t"], h_raw["omega_ref"], color=CR,  lw=0.8, label="ω* raw")
    ax.plot(h_raw["t"], h_raw["w_meas"],    color=CR2, lw=0.8, ls=":", label="ω meas raw")
    ax.plot(h_cln["t"], h_cln["omega_ref"], color=CC,  lw=1.2, ls="--", label="ω* clean")
    ax.plot(h_cln["t"], h_cln["w_meas"],    color=CC2, lw=1.0, ls="-.", label="ω meas clean")
    fmt(ax, "Yaw rate tracking", "Time (s)", "ω (rad/s)"); ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(h_raw["t"], h_raw["theta1"], color=CR, lw=0.8, alpha=0.85, label="θ₁ raw")
    ax.plot(h_cln["t"], h_cln["theta1"], color=CC, lw=1.2, ls="--", label="θ₁ clean")
    ax.axhline(15, color="grey", ls=":", lw=0.8); ax.axhline(-15, color="grey", ls=":", lw=0.8)
    fmt(ax, "Pad tilt θ₁ (pad 1 CCW)", "Time (s)", "θ₁ (°)"); ax.legend()

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(h_raw["t"], h_raw["theta2"], color=CR, lw=0.8, alpha=0.85, label="θ₂ raw")
    ax.plot(h_cln["t"], h_cln["theta2"], color=CC, lw=1.2, ls="--", label="θ₂ clean")
    ax.axhline(15, color="grey", ls=":", lw=0.8); ax.axhline(-15, color="grey", ls=":", lw=0.8)
    fmt(ax, "Pad tilt θ₂ (pad 2 CW)", "Time (s)", "θ₂ (°)"); ax.legend()

    # # Row 2 — zoomed first 5s
    # def zoom(gs_pos, tr, yr, tc, yc, title, ylabel, tlim=5.0):
    #     tr2 = [v for v in tr if v <= tlim]; yr2 = yr[:len(tr2)]
    #     tc2 = [v for v in tc if v <= tlim]; yc2 = yc[:len(tc2)]
    #     ax  = fig.add_subplot(gs_pos)
    #     ax.plot(tr2, yr2, color=CR, lw=0.8, alpha=0.9, label="Raw")
    #     ax.plot(tc2, yc2, color=CC, lw=1.4, ls="--",   label="Clean")
    #     fmt(ax, title + " (first 5 s)", "Time (s)", ylabel); ax.legend(fontsize=6.5)

    # zoom(gs[2,0], h_raw["t"],h_raw["v_ref"],    h_cln["t"],h_cln["v_ref"],    "v* reference", "v* (m/s)")
    # zoom(gs[2,1], h_raw["t"],h_raw["omega_ref"], h_cln["t"],h_cln["omega_ref"],"ω* reference", "ω* (rad/s)")
    # zoom(gs[2,2], h_raw["t"],h_raw["theta1"],    h_cln["t"],h_cln["theta1"],   "Pad tilt θ₁",  "θ₁ (°)")
    # zoom(gs[2,3], h_raw["t"],h_raw["theta2"],    h_cln["t"],h_cln["theta2"],   "Pad tilt θ₂",  "θ₂ (°)")

    plt.savefig("snail_simulation_new.png", dpi=300, bbox_inches="tight")
    print("Plot saved → snail_simulation_new.png")
    plt.show()

    # export_trials(
    #     "snail_trials.npz",
    #     waypoints,
    #     thetas=[0.0, 15.0, 90.0],
    #     n_runs=8,
    # )
    # print("Trials saved → snail_trials.npz")
