"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.accel_controller import AccelController
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  ECO, NORMAL, SPORT, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, RISE_RATE, \
  STOCK_A_CRUISE_MAX_V, STOCK_RISE_RATE, HARD_BRAKE_TARGET_ACCEL, OVERBITE_CAP, \
  STOP_PASSTHROUGH_V, AccelerationPersonality

T_IDXS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
_EPS = 1e-6


class FakeParams:
  def __init__(self, store=None):
    self.store = dict(store or {})

  def get_bool(self, key):
    return bool(self.store.get(key, False))

  def get(self, key, return_default=False):
    return int(self.store.get(key, 1))

  def put(self, key, val, block=False):
    self.store[key] = val


def make_sm(v_ego=20.0, lead_status=False, lead_d=0.0, lead_vlead=0.0):
  lead = SimpleNamespace(status=lead_status, dRel=lead_d, vLead=lead_vlead)
  return {'carState': SimpleNamespace(vEgo=v_ego), 'radarState': SimpleNamespace(leadOne=lead)}


def make_controller(enabled=True, personality=NORMAL, crash_cnt=0):
  store = {"AccelPersonalityEnabled": enabled, "AccelPersonality": int(personality)}
  ctrl = AccelController(CP=SimpleNamespace(), mpc=SimpleNamespace(crash_cnt=crash_cnt), params=FakeParams(store))
  ctrl.update(make_sm())
  return ctrl


def flat_traj(value):
  return [float(value)] * len(T_IDXS)


# --- Profiles / off==stock ---------------------------------------------------

def test_enum_source_parity():
  assert (ECO, NORMAL, SPORT) == (AccelerationPersonality.eco, AccelerationPersonality.normal, AccelerationPersonality.sport)
  assert (PERSONALITY_MIN, PERSONALITY_MAX) == (0, 2)


def test_disabled_forces_normal_and_stock_ceiling():
  ctrl = make_controller(enabled=False, personality=SPORT)
  assert ctrl.personality() == NORMAL
  assert not ctrl.enabled()
  for v in (0.0, 10.0, 25.0, 40.0):
    assert ctrl.get_max_accel(v) == pytest.approx(np.interp(v, A_CRUISE_MAX_BP, STOCK_A_CRUISE_MAX_V))
  assert ctrl.get_rise_rate() == STOCK_RISE_RATE


def test_disabled_passes_brake_through():
  ctrl = make_controller(enabled=False)
  for raw in (-3.0, -1.5, -0.5, 0.0, 1.0):
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    assert out == pytest.approx(raw, abs=_EPS)


def test_normal_is_distinct_from_stock():
  # off==stock is enforced via the disabled path, NOT by NORMAL==stock, so enabled NORMAL is free to differ.
  ctrl = make_controller(personality=NORMAL)
  assert ctrl.get_max_accel(0.0) != pytest.approx(np.interp(0.0, A_CRUISE_MAX_BP, STOCK_A_CRUISE_MAX_V))
  assert ctrl.get_rise_rate() != STOCK_RISE_RATE


def test_ceiling_ordering_eco_lt_normal_lt_sport():
  eco, normal, sport = (make_controller(personality=p) for p in (ECO, NORMAL, SPORT))
  for v in (0.0, 14.0, 25.0, 40.0):
    assert eco.get_max_accel(v) < normal.get_max_accel(v) < sport.get_max_accel(v)
  assert eco.get_rise_rate() < normal.get_rise_rate() < sport.get_rise_rate()


def test_rise_rate_ordering():
  assert RISE_RATE[ECO] < RISE_RATE[NORMAL] < RISE_RATE[SPORT]


# --- SAFETY: never weaker than the plan, hard brakes never delayed --------------

@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_never_weaker_than_plan_sustained(personality):
  # Core safety invariant: on the brake side the output is NEVER weaker than the plan (only equal or deeper).
  ctrl = make_controller(personality=personality)
  for raw in [0.0, -0.2, -0.5, -0.9, -1.2, -1.5, -2.0] + [-2.0] * 20:
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    if raw < 0.0:
      assert out <= raw + _EPS


@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_never_weaker_random_walk(personality):
  rng = np.random.default_rng(0)
  ctrl = make_controller(personality=personality)
  for _ in range(500):
    raw = float(rng.uniform(-2.5, 1.5))
    traj = flat_traj(raw - float(rng.uniform(0.0, 0.6)))
    out = ctrl.smooth_target_accel(raw, traj, T_IDXS, should_stop=False)
    if raw < 0.0:
      assert out <= raw + _EPS


@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_hard_brake_passes_through_immediately(personality):
  # Regression for route 00000466 near-crash: a sudden hard brake (plan steps deep) must reach FULL depth
  # on the FIRST frame -- never rate-limited / delayed, or the car under-brakes into a closing lead.
  ctrl = make_controller(personality=personality)
  out = ctrl.smooth_target_accel(-3.5, flat_traj(-3.5), T_IDXS, should_stop=False)
  assert out == pytest.approx(-3.5, abs=_EPS)
  assert ctrl.bypassed()


def test_sudden_lead_no_brake_delay():
  # The exact 466 shape: cruising (plan +1.7, no brake) then a fast lead appears and the plan steps to max
  # brake. The commanded brake must hit full depth immediately, not ramp in over time.
  ctrl = make_controller(personality=ECO)
  for _ in range(5):
    ctrl.smooth_target_accel(1.7, flat_traj(1.7), T_IDXS, should_stop=False)   # cruising, no lead
  out = ctrl.smooth_target_accel(-3.5, flat_traj(-3.5), T_IDXS, should_stop=False)  # lead appears
  assert out == pytest.approx(-3.5, abs=_EPS)                                  # full brake, zero delay


def test_should_stop_passes_through():
  ctrl = make_controller(personality=ECO)
  out = ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=True)
  assert out == pytest.approx(-1.0, abs=_EPS)
  assert ctrl.bypassed()


def test_fcw_crash_passes_through():
  ctrl = make_controller(personality=ECO, crash_cnt=3)
  out = ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=False)
  assert out == pytest.approx(-1.0, abs=_EPS)
  assert ctrl.bypassed()


def test_blended_never_weaker():
  # Blended/e2e (stock_brake): never weaker than the plan (may anticipate via the never-weaker front-load).
  ctrl = make_controller(personality=ECO)
  for raw in [0.0, -0.3, -0.6, -0.9, -1.0, -1.0, -1.0]:
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False, stock_brake=True)
    assert out <= raw + _EPS


# --- Anticipatory front-load (never weaker, capped) ------------------------------

def test_front_load_brakes_before_plan():
  # A deeper brake is predicted ahead (brake_need=1.0) while the live plan is still flat -> front-load
  # brakes early (output goes negative), but the smooth branch keeps it never weaker than the plan.
  ctrl = make_controller(personality=ECO)
  out = ctrl.smooth_target_accel(0.0, flat_traj(-1.0), T_IDXS, should_stop=False)
  assert out < 0.0
  assert ctrl.smooth_active()
  assert ctrl.brake_need() == pytest.approx(1.0)


def test_front_load_anticipates_below_live_plan():
  # When the live plan is gently braking and a deeper brake is predicted, the front-load deepens below the
  # live plan (anticipatory early brake), settling within OVERBITE_CAP of it.
  ctrl = make_controller(personality=ECO)
  out = 0.0
  for _ in range(20):
    out = ctrl.smooth_target_accel(-0.2, flat_traj(-1.5), T_IDXS, should_stop=False)
  assert out < -0.2 - _EPS                                   # deeper than the live -0.2 plan
  assert out >= -0.2 - OVERBITE_CAP - _EPS                   # but never more than the cap below it


def test_overbite_cap_limits_frontload_vs_live_plan():
  # Cut-in/merge: plan still wants throttle (+0.5) while a deep brake is predicted -> front-load may not
  # settle more than OVERBITE_CAP below the live plan (no abrupt early over-bite).
  ctrl = make_controller(personality=ECO)
  traj = [0.5, 0.3, 0.0, -0.5, -1.5, -2.0] + [-2.0] * (len(T_IDXS) - 6)
  out = 0.0
  for _ in range(10):
    out = ctrl.smooth_target_accel(0.5, traj, T_IDXS, should_stop=False)
  assert ctrl.smooth_active()
  assert out == pytest.approx(0.5 - OVERBITE_CAP, abs=1e-3)


# --- Stop / low-speed neutrality -------------------------------------------------

def test_low_speed_brake_is_stock_passthrough():
  # Stop/creep regime (vEgo < STOP_PASSTHROUGH_V): braking is stock so the stop distance matches OFF.
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=STOP_PASSTHROUGH_V - 0.1))
  for raw in (-0.3, -1.0):
    out = ctrl.smooth_target_accel(raw, flat_traj(-1.5), T_IDXS, should_stop=False)
    assert out == pytest.approx(raw, abs=_EPS)
    assert not ctrl.smooth_active()


def test_low_speed_launch_still_shapes():
  # The low-speed brake passthrough must NOT neutralize positive-accel (launch) shaping.
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=STOP_PASSTHROUGH_V - 0.1))
  ctrl.smooth_target_accel(0.0, flat_traj(0.0), T_IDXS, should_stop=False)
  out = ctrl.smooth_target_accel(1.5, flat_traj(1.5), T_IDXS, should_stop=False)
  assert out < 1.5                                           # rise-rate limited (shaped)


def test_stop_imminent_passthrough_but_moving_follow_shapes():
  # Stop coming (plan speed -> ~0): stock passthrough (no coast-in). Slowing to a moving follow: front-load
  # stays active so the early-brake goal is preserved.
  ctrl = make_controller(personality=ECO)
  stopping = [3.0, 2.0, 1.0, 0.4, 0.0] + [0.0] * (len(T_IDXS) - 5)
  out = ctrl.smooth_target_accel(-0.1, flat_traj(-1.0), T_IDXS, should_stop=False, speed_trajectory=stopping)
  assert not ctrl.smooth_active()
  assert out == pytest.approx(-0.1, abs=_EPS)
  moving = [8.0] * len(T_IDXS)
  ctrl.smooth_target_accel(-0.1, flat_traj(-1.0), T_IDXS, should_stop=False, speed_trajectory=moving)
  assert ctrl.smooth_active()


def test_stop_enforce_brakes_when_mpc_creeps_inside_target():
  # Crawl behind a stopped lead inside the target gap: the stock plan eases off (~ -0.1), but the enforcer
  # holds a gentle decel to stop at the target -> output is DEEPER than the easing plan (no creep-in).
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=1.5, lead_status=True, lead_d=4.0, lead_vlead=0.0))   # inside 5.5m target, moving
  out = ctrl.smooth_target_accel(-0.1, flat_traj(-0.1), T_IDXS, should_stop=False)
  assert out < -0.1 - _EPS                              # enforcer added braking vs the easing plan
  assert out >= -2.0 - _EPS                             # but gentle (capped), never a grab


def test_stop_enforce_off_when_disabled():
  # Disabled controller: enforcer is a no-op (off == stock).
  ctrl = make_controller(enabled=False, personality=ECO)
  ctrl.update(make_sm(v_ego=1.5, lead_status=True, lead_d=4.0, lead_vlead=0.0))
  out = ctrl.smooth_target_accel(-0.1, flat_traj(-0.1), T_IDXS, should_stop=False)
  assert out == pytest.approx(-0.1, abs=_EPS)


def test_stop_enforce_no_op_past_target_and_moving_lead():
  # Past the target gap (lead far): no enforcement. Moving lead: no enforcement (only near-stopped leads).
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=1.5, lead_status=True, lead_d=12.0, lead_vlead=0.0))   # far -> no floor
  assert ctrl.smooth_target_accel(-0.1, flat_traj(-0.1), T_IDXS, should_stop=False) == pytest.approx(-0.1, abs=_EPS)
  ctrl.update(make_sm(v_ego=1.5, lead_status=True, lead_d=4.0, lead_vlead=4.0))    # moving lead -> no floor
  assert ctrl.smooth_target_accel(-0.1, flat_traj(-0.1), T_IDXS, should_stop=False) == pytest.approx(-0.1, abs=_EPS)


def test_stop_enforce_never_weaker():
  # The enforcer only ever ADDS braking: output is never weaker than the plan.
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=4.5, lead_vlead=0.0))
  for raw in (-0.05, -0.3, -1.0, -2.5):
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    assert out <= raw + _EPS


def test_disabled_hard_brake_is_instant_stock():
  ctrl = make_controller(enabled=False, personality=ECO)
  out = ctrl.smooth_target_accel(-3.0, flat_traj(-3.0), T_IDXS, should_stop=False)
  assert out == pytest.approx(-3.0, abs=_EPS)


# --- Misc ------------------------------------------------------------------------

def test_out_of_range_personality_clamps():
  ctrl = AccelController(CP=SimpleNamespace(), mpc=SimpleNamespace(crash_cnt=0),
                         params=FakeParams({"AccelPersonalityEnabled": True, "AccelPersonality": 99}))
  ctrl.update(make_sm())
  assert ctrl.personality() == PERSONALITY_MAX


def test_reset_passes_through():
  ctrl = make_controller(personality=ECO)
  out = ctrl.smooth_target_accel(0.0, flat_traj(-1.0), T_IDXS, should_stop=False, reset=True)
  assert out == pytest.approx(0.0, abs=_EPS)
  assert not ctrl.bypassed()
