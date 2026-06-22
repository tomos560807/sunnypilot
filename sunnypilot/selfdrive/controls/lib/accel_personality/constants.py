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

# Accel ceiling + its upward slew rate (the POSITIVE-accel / launch + cruise-accel side; independent of
# braking, so tuning here does NOT touch the gentle-brake goals). off==stock is enforced in accel_controller
# (get_max_accel/get_rise_rate fall back to STOCK_* when disabled), so the NORMAL profile is free to differ
# from stock -- all three tiers are now distinct. Start-from-stop is FAST: launch peak (v=0) is firm and the
# rise rate (how fast the ceiling opens) is well above stock 0.05 in every tier, stepped ECO < NORMAL < SPORT.
A_CRUISE_MAX_BP = [0., 14., 25., 40.]
STOCK_A_CRUISE_MAX_V = [1.6, 0.7, 0.2, 0.08]
STOCK_RISE_RATE = 0.05
A_CRUISE_MAX_V = {
  ECO:    [1.70, 0.75, 0.25, 0.10],   # prompt launch, efficient cruise
  NORMAL: [2.10, 1.10, 0.50, 0.18],   # quick launch, balanced cruise
  SPORT:  [2.60, 1.55, 0.85, 0.35],   # fast launch, strong cruise
}
RISE_RATE = {ECO: 0.10, NORMAL: 0.15, SPORT: 0.22}   # ceiling open-rate: all >> stock 0.05 for fast take-off

# Anticipatory front-load: predicted brake need (m/s^2) -> early decel target (m/s^2). When the 3s plan
# lookahead predicts a brake, start a gentle decel EARLY so braking is spread out instead of arriving as
# one late firm onset (route 00000456). It is one-sided: min(., raw) keeps the output NEVER weaker than the
# plan, so it can only brake EQUAL or EARLIER/DEEPER, never softer. Hard brakes (brake_need>=HARD_BRAKE_NEED
# or raw<=HARD_BRAKE_TARGET_ACCEL) pass straight through at full strength.
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

# Front-load over-bite cap. The SMOOTH_DECEL front-load is allowed to brake at most this much DEEPER than
# the live raw plan. On a cut-in/merge the 3s brake_need spikes and the table would front-load a firm brake
# while the plan still wants throttle/coast -> an abrupt early bite (the "worse on merge" feel; routes
# 45e/460). This binds only in that contradictory case; once the plan itself brakes, raw-OVERBITE_CAP sits
# below the table value so the table wins and the anticipatory early brake (route 456 fix) is preserved.
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

# Stop/creep stop-neutrality. Below this ego speed, the brake side hands the plan straight through (stock),
# so the controller cannot soften the final crawl and let the car coast in closer than stock. Matches the
# radar_distance low-speed stop-neutrality so ON == OFF near stops. Positive-accel (launch) shaping is
# unaffected (the launch profiles still apply via the accel ceiling).
STOP_PASSTHROUGH_V = 5.0          # m/s ego speed below which braking is stock passthrough
