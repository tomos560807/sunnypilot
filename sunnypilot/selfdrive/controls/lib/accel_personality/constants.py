"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom

AccelerationPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ECO = AccelerationPersonality.eco
NORMAL = AccelerationPersonality.normal
SPORT = AccelerationPersonality.sport

PERSONALITY_MIN = min(AccelerationPersonality.schema.enumerants.values())
PERSONALITY_MAX = max(AccelerationPersonality.schema.enumerants.values())

# Positive-accel ceiling + its upward slew rate (launch/cruise side; independent of braking). off==stock is
# enforced in accel_controller (falls back to STOCK_* when disabled), so the tiers are free to differ.
A_CRUISE_MAX_BP = [0., 14., 25., 40.]
STOCK_A_CRUISE_MAX_V = [1.6, 0.7, 0.2, 0.08]
STOCK_RISE_RATE = 0.05
A_CRUISE_MAX_V = {
  ECO:    [1.70, 0.75, 0.25, 0.10],   # prompt launch, efficient cruise
  NORMAL: [2.10, 1.10, 0.50, 0.18],   # quick launch, balanced cruise
  SPORT:  [2.60, 1.55, 0.85, 0.35],   # fast launch, strong cruise
}
RISE_RATE = {ECO: 0.10, NORMAL: 0.15, SPORT: 0.22}   # ceiling open-rate: all >> stock 0.05 for fast take-off

# Anticipatory front-load: predicted brake need (m/s^2) -> early decel target (m/s^2). Starts a gentle
# decel early when a brake is predicted, so it arrives spread out, not as one late firm onset. One-sided
# (never weaker than the plan).
SMOOTH_DECEL_BP = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
SMOOTH_DECEL_V = {
  ECO:    [0.00, -0.08, -0.20, -0.35, -0.55, -0.78, -1.00],
  NORMAL: [0.00, -0.13, -0.30, -0.55, -0.84, -1.12, -1.40],
  SPORT:  [0.00, -0.17, -0.40, -0.72, -1.05, -1.35, -1.65],
}
BRAKE_DEEPENING_JERK = {ECO: 0.5, NORMAL: 0.8, SPORT: 1.0}
BRAKE_RELEASE_JERK = 2.0
ACCEL_RISE_JERK = {ECO: 1.0, NORMAL: 1.5, SPORT: 2.2}   # accel-onset jerk: higher = snappier take-off, stepped per tier

SMOOTH_DECEL_LOOKAHEAD_T = 3.0
MIN_SMOOTH_BRAKE_NEED = 0.2

# Cap how much DEEPER than the live plan the front-load may bite -> no abrupt over-bite on a cut-in
# brake_need spike (binds only when the plan still wants throttle; once it brakes, the table wins).
OVERBITE_CAP = 0.30   # m/s^2 max front-load depth below the live plan

# Hard brake: at/below this accel, or this predicted brake_need within the lookahead, the controller hands
# the plan straight through at full strength and rate (no front-load, no rate limit) -- a firm/closing-lead
# brake must never be delayed, softened or rate-limited.
HARD_BRAKE_TARGET_ACCEL = -1.5
HARD_BRAKE_NEED = 2.6

# Stop-imminent stand-down. When the plan predicts a near-stop within the lookahead, hand the plan straight
# through (stock decel) so the car stops at the proper gap with no front-load coast-in. Keyed on the
# PREDICTED speed reaching ~0 (covers lead AND light/sign stops), not raw ego speed.
STOP_IMMINENT_VEGO = 1.0          # m/s  plan-predicted speed below this within the lookahead == stop coming
STOP_IMMINENT_LOOKAHEAD_T = 3.0   # s

# Below this ego speed the brake side is stock passthrough, so stop distance is byte-identical to off.
STOP_PASSTHROUGH_V = 5.0          # m/s

# Low-speed stop-distance enforcer. The stock MPC loses gap-cost leverage at crawl and creeps inside
# STOP_DISTANCE behind a stopped lead. This is a never-weaker floor: command the gentle decel that brings
# the car to rest at STOP_ENFORCE_DIST and take min(plan, floor) -> only ever adds braking, self-targeting.
STOP_ENFORCE_V = 5.0              # m/s: only enforce at/below this ego speed
STOP_ENFORCE_DIST = 5.5          # m: target standstill gap (under STOP_DISTANCE=6 for the radar rear-of-lead offset)
STOP_ENFORCE_RANGE = 3.0         # m: only within DIST+RANGE of the lead (final-approach creep zone)
STOP_ENFORCE_LEAD_V = 1.5        # m/s: only behind a near-stopped lead
STOP_ENFORCE_MAX_DECEL = -1.8    # m/s^2: cap -> always a gentle hold, never a grab
STOP_ENFORCE_MIN_GAP = 0.5       # m: kinematic denominator floor
