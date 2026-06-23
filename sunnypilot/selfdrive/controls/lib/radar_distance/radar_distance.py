"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

RadarDistance: smooths the lead the longitudinal MPC follows, without ever reporting a farther-or-faster
lead than reality (so braking is always >= stock, never weaker). Three jobs, all above LOW_SPEED_PASSTHROUGH_V:
  - flicker-hold: keep a just-dropped, recently-sustained lead alive through a brief radar dropout.
  - lead-speed smoothing: lag the lead ACCELERATING (damps the catch-up surge -> less rubber-band) while
    passing the lead SLOWING straight through (instant brake).
At/below LOW_SPEED_PASSTHROUGH_V (stop/creep) it returns the raw radarstate unchanged -> byte-stock stops.
Default off => stock passthrough.
"""

from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

HOLD_MAX_FRAMES = 10        # ~0.5s flicker-hold cap, since the last sustained lead
SUSTAIN_FRAMES = 2          # consecutive valid frames to arm the hold
DROPOUT_DREL = 1.0
FCW_PROB_CAP = 0.9          # held lead can't reach the FCW gate (>0.9)
MIN_HELD_DREL = 0.5

# Stop/creep regime: return the raw radarstate so stop distance is byte-identical to stock (off==on).
LOW_SPEED_PASSTHROUGH_V = 5.0   # m/s

# Lead-speed smoothing: time constant for lagging a lead that is speeding up. Falls are instant, so the
# reported vLead is always <= real -> obstacle is never farther than stock -> braking never weaker.
VLEAD_RISE_TAU = 1.0            # s
_VLEAD_RISE_ALPHA = DT_MDL / VLEAD_RISE_TAU


class _LeadView:
  # Mirror of a lead with a smoothed vLead (<= real). Used above the stop gate to damp the catch-up surge.
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb')

  def __init__(self, src, vlead):
    self.status = src.status
    self.dRel = src.dRel
    self.yRel = src.yRel
    self.vRel = src.vRel
    self.vLead = vlead
    self.vLeadK = vlead
    self.aLeadK = src.aLeadK
    self.aLeadTau = src.aLeadTau
    self.modelProb = src.modelProb


class _HeldLead:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb')

  def __init__(self, dRel, vRel, vLead, aLeadK, aLeadTau, modelProb):
    self.status = True
    self.dRel = dRel
    self.vRel = vRel
    self.vLead = vLead
    self.vLeadK = vLead
    self.aLeadK = aLeadK
    self.aLeadTau = aLeadTau
    self.modelProb = modelProb
    self.yRel = 0.0


class _RadarStateProxy:
  __slots__ = ('leadOne', 'leadTwo')

  def __init__(self, lead_one, lead_two):
    self.leadOne = lead_one
    self.leadTwo = lead_two


class _LeadHold:
  def __init__(self):
    self._last = None
    self._sustained = 0
    self._since_real = 0
    self._armed = False
    self._held_dRel = 0.0
    self._vlead_f = None        # smoothed vLead (lag-up / instant-down)

  def reset(self):
    self.__init__()

  def step(self, raw):
    # Validity mirrors the MPC (keys off status alone). modelProb is NOT a gate: radard's low_speed_override
    # emits a real close lead with modelProb=0.0, so gating on prob dropped real stop-and-go leads.
    if raw.status and raw.dRel > DROPOUT_DREL:
      self._last = (raw.dRel, raw.vRel, raw.vLead, raw.aLeadK, raw.aLeadTau, raw.modelProb)
      self._sustained += 1
      if self._sustained >= SUSTAIN_FRAMES:
        self._since_real = 0
        self._armed = True
      return raw

    self._sustained = 0
    self._since_real += 1
    if self._armed and self._last is not None and self._since_real <= HOLD_MAX_FRAMES:
      dRel0, vRel0, vLead0, aLeadK0, aLeadTau0, prob0 = self._last
      if self._since_real == 1:
        self._held_dRel = dRel0
      self._held_dRel = max(MIN_HELD_DREL, self._held_dRel - max(-vRel0, 0.0) * DT_MDL)
      return _HeldLead(self._held_dRel, vRel0, vLead0, min(aLeadK0, 0.0), aLeadTau0, min(prob0, FCW_PROB_CAP))

    self._armed = False
    return raw

  def smooth_vlead(self, lead):
    # Lag the lead speeding up; pass slowing through instantly. Reported vLead stays <= real => the MPC's
    # obstacle is never farther than stock (never-weaker), and the catch-up surge to a fidgety lead is damped.
    if not lead.status:
      self._vlead_f = None
      return lead
    v = float(lead.vLead)
    if self._vlead_f is None or v <= self._vlead_f:
      self._vlead_f = v                                      # instant on slow-down / first sample
      return lead
    self._vlead_f += (v - self._vlead_f) * _VLEAD_RISE_ALPHA  # lag on speed-up
    return _LeadView(lead, self._vlead_f)


class RadarDistanceController:
  def __init__(self, CP: structs.CarParams, params=None):
    self._CP = CP
    self._params = params or Params()
    self._frame = 0
    self._v_ego = 0.0
    self._enabled = self._params.get_bool("RadarDistance")
    self._one = _LeadHold()
    self._two = _LeadHold()

  def _read_params(self) -> None:
    enabled = self._params.get_bool("RadarDistance")
    if enabled and not self._enabled:
      self._one.reset()
      self._two.reset()
    self._enabled = enabled

  def update(self, sm) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._read_params()
    self._v_ego = float(sm['carState'].vEgo)
    self._frame += 1

  def enabled(self) -> bool:
    return self._enabled

  def smooth_radarstate(self, radarstate):
    if not self._enabled:
      return radarstate
    one = self._one.step(radarstate.leadOne)
    two = self._two.step(radarstate.leadTwo)
    if self._v_ego < LOW_SPEED_PASSTHROUGH_V:               # stop/creep -> raw (byte-stock stops)
      return radarstate
    return _RadarStateProxy(self._one.smooth_vlead(one), self._two.smooth_vlead(two))
