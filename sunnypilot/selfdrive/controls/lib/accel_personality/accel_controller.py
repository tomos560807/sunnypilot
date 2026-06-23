"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Acceleration personality: per-profile launch/cruise accel ceiling (ECO/NORMAL/SPORT) plus an
anticipatory brake front-load. SAFETY INVARIANT: on the brake side the output is NEVER WEAKER than the
plan -- it can only be EQUAL or DEEPER (front-load). It never softens, delays, or rate-limits a brake,
so it can never under-brake a closing lead. Hard brakes, stops and low speed pass the plan straight
through (stock). Disabled => byte-stock.
"""

from collections.abc import Sequence

import numpy as np

from cereal import messaging
from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  NORMAL, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, A_CRUISE_MAX_V, RISE_RATE, \
  STOCK_A_CRUISE_MAX_V, STOCK_RISE_RATE, SMOOTH_DECEL_BP, SMOOTH_DECEL_V, BRAKE_DEEPENING_JERK, \
  BRAKE_RELEASE_JERK, ACCEL_RISE_JERK, SMOOTH_DECEL_LOOKAHEAD_T, MIN_SMOOTH_BRAKE_NEED, \
  HARD_BRAKE_TARGET_ACCEL, HARD_BRAKE_NEED, OVERBITE_CAP, STOP_PASSTHROUGH_V, \
  STOP_IMMINENT_VEGO, STOP_IMMINENT_LOOKAHEAD_T, \
  STOP_ENFORCE_V, STOP_ENFORCE_DIST, STOP_ENFORCE_RANGE, STOP_ENFORCE_LEAD_V, STOP_ENFORCE_MAX_DECEL, STOP_ENFORCE_MIN_GAP

_ZERO_ACCEL_EPS = 1e-6


class AccelController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._frame = 0
    self._enabled = self._params.get_bool("AccelPersonalityEnabled")
    self._personality = NORMAL
    self._v_ego = 0.0
    self._last_target_accel = 0.0
    self._brake_need = 0.0
    self._decel_target = 0.0
    self._smooth_active = False
    self._bypassed = False
    self._lead_status = False
    self._lead_d = 0.0
    self._lead_vlead = 0.0
    self._read_params()

  def _read_params(self) -> None:
    self._enabled = self._params.get_bool("AccelPersonalityEnabled")
    if not self._enabled:
      self._personality = NORMAL
      return
    self._personality = get_sanitize_int_param("AccelPersonality", PERSONALITY_MIN, PERSONALITY_MAX, self._params)

  def update(self, sm: messaging.SubMaster) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._read_params()
    self._v_ego = sm['carState'].vEgo
    lead = sm['radarState'].leadOne          # raw radard lead (== what the MPC sees at crawl, where the enforcer acts)
    self._lead_status = bool(lead.status)
    self._lead_d = float(lead.dRel)
    self._lead_vlead = float(lead.vLead)
    self._frame += 1

  def get_max_accel(self, v_ego: float) -> float:
    # Disabled -> stock ceiling (off == stock, independent of the NORMAL profile so NORMAL is free to differ).
    table = A_CRUISE_MAX_V[self._personality] if self._enabled else STOCK_A_CRUISE_MAX_V
    return float(np.interp(v_ego, A_CRUISE_MAX_BP, table))

  def get_rise_rate(self) -> float:
    # Disabled -> stock rise rate (off == stock, independent of the NORMAL profile).
    return RISE_RATE[self._personality] if self._enabled else STOCK_RISE_RATE

  def get_decel_target(self, brake_need: float) -> float:
    return float(np.interp(max(0.0, float(brake_need)), SMOOTH_DECEL_BP, SMOOTH_DECEL_V[self._personality]))

  def smooth_target_accel(self, raw_target_accel: float, accel_trajectory: Sequence[float], t_idxs: Sequence[float],
                          should_stop: bool, reset: bool = False, stock_brake: bool = False,
                          speed_trajectory: Sequence[float] | None = None) -> float:
    raw = float(raw_target_accel)
    self._brake_need = self._compute_brake_need(raw, accel_trajectory, t_idxs)
    self._decel_target = 0.0
    self._smooth_active = False
    self._bypassed = False

    out = self._shape(raw, should_stop, reset, speed_trajectory, t_idxs)
    out = self._stop_enforce(out)   # never-weaker low-speed floor: no creep inside the target stop gap
    return self._finalize(out)

  def _shape(self, raw: float, should_stop: bool, reset: bool, speed_trajectory, t_idxs) -> float:
    # --- Full stock passthroughs (output is exactly the plan, no shaping) ---
    if reset or not self._enabled:
      return raw                                               # disabled / reset
    if self._v_ego < STOP_PASSTHROUGH_V and raw <= 0.0:
      return raw                                               # stop/creep regime: braking is stock (no coast-in)
    self._bypassed = self._emergency_bypass(raw, should_stop)
    if self._bypassed or self._stop_imminent(speed_trajectory, t_idxs):
      return raw                                               # hard brake / coming stop: full strength, no delay

    # Anticipatory front-load, capped at OVERBITE_CAP below the live plan (avoids an abrupt over-bite on a
    # cut-in brake_need spike). min(., raw) keeps the output never weaker than the plan -> never under-brakes.
    target = raw
    if self._brake_need >= MIN_SMOOTH_BRAKE_NEED:
      self._smooth_active = True
      self._decel_target = max(self.get_decel_target(self._brake_need), raw - OVERBITE_CAP)
      target = min(raw, self._decel_target)
    slewed = self._slew(target)
    return min(slewed, raw) if raw < 0.0 else slewed

  def _stop_enforce(self, out: float) -> float:
    # Never-weaker low-speed floor: bring the car to rest at STOP_ENFORCE_DIST behind a near-stopped lead,
    # so the stock MPC's crawl-creep cannot park us inside the target gap. Disabled => no-op (off==stock).
    if not (self._enabled and self._lead_status and 0.1 < self._v_ego < STOP_ENFORCE_V
            and self._lead_vlead < STOP_ENFORCE_LEAD_V
            and 0.1 < self._lead_d < STOP_ENFORCE_DIST + STOP_ENFORCE_RANGE):   # only the final-approach creep zone
      return out
    gap = self._lead_d - STOP_ENFORCE_DIST                    # distance left before reaching the target gap
    floor = -(self._v_ego ** 2) / (2.0 * max(gap, STOP_ENFORCE_MIN_GAP))   # gentle decel to stop at the target
    floor = max(floor, STOP_ENFORCE_MAX_DECEL)                # cap -> gentle hold, never a grab
    return min(out, floor)                                    # never weaker than the plan

  def _stop_imminent(self, speed_trajectory: Sequence[float] | None, t_idxs: Sequence[float]) -> bool:
    # plan predicts a near-stop within the lookahead -> a stop is coming (lead or light/sign).
    if speed_trajectory is None:
      return False
    return any(float(s) < STOP_IMMINENT_VEGO
               for s, t in zip(speed_trajectory, t_idxs, strict=False) if float(t) <= STOP_IMMINENT_LOOKAHEAD_T)

  def _compute_brake_need(self, raw_target_accel: float, accel_trajectory: Sequence[float], t_idxs: Sequence[float]) -> float:
    min_accel = float(raw_target_accel)
    for accel, t in zip(accel_trajectory, t_idxs, strict=False):
      if float(t) <= SMOOTH_DECEL_LOOKAHEAD_T:
        min_accel = min(min_accel, float(accel))
    return max(0.0, -min_accel)

  def _emergency_bypass(self, raw_target_accel: float, should_stop: bool) -> bool:
    return (self._mpc.crash_cnt > 0 or should_stop or
            raw_target_accel <= HARD_BRAKE_TARGET_ACCEL or self._brake_need >= HARD_BRAKE_NEED)

  def _slew(self, target_accel: float) -> float:
    # Jerk-limit the brake DEEPENING (smooths the front-load's extra depth). On the brake side the caller
    # clamps with min(., raw), so this NEVER delays a real brake -- when the plan is deeper than the slewed
    # value, min(.) picks the plan and the brake passes through at full rate.
    target_accel = float(target_accel)
    if target_accel <= self._last_target_accel:
      jmax = BRAKE_DEEPENING_JERK[self._personality]
      return self._clean_accel(max(target_accel, self._last_target_accel - jmax * DT_MDL))
    return self._slew_up(target_accel)

  def _slew_up(self, target_accel: float) -> float:
    # Releasing the brake / accelerating: rate-limit the rise (release jerk on the brake side, the
    # personality accel-rise jerk on the throttle side).
    if self._last_target_accel < 0.0:
      released = min(target_accel, self._last_target_accel + BRAKE_RELEASE_JERK * DT_MDL)
      if released <= 0.0:
        return self._clean_accel(released)
      return self._clean_accel(min(target_accel, ACCEL_RISE_JERK[self._personality] * DT_MDL))
    step = ACCEL_RISE_JERK[self._personality] * DT_MDL
    return self._clean_accel(min(target_accel, self._last_target_accel + step))

  def _finalize(self, target_accel: float) -> float:
    target_accel = self._clean_accel(target_accel)
    self._last_target_accel = target_accel
    return target_accel

  @staticmethod
  def _clean_accel(accel: float) -> float:
    accel = float(accel)
    return 0.0 if abs(accel) < _ZERO_ACCEL_EPS else accel

  def enabled(self) -> bool:
    return self._enabled

  def personality(self):
    return self._personality

  def max_accel(self) -> float:
    return self.get_max_accel(self._v_ego)

  def brake_need(self) -> float:
    return self._brake_need

  def decel_target(self) -> float:
    return self._decel_target

  def smooth_active(self) -> bool:
    return self._smooth_active

  def bypassed(self) -> bool:
    return self._bypassed
