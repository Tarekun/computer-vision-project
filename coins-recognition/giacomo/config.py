"""Physical constants and frozen calibration for the giacomo pipeline.

Everything giacomo shares with tiziano (physics, detection, canonical-crop
geometry, shape/colour cue calibration, most of the fusion-weight block) lives
in common/config.py. This file only holds what's actually specific to
giacomo: the pairwise/same-class size-term weights, which tiziano recalibrates
to a different value (0.5/0.2 there vs 0.25/0.15 here).
"""
from common.config import *  # noqa: F401,F403

W_ORDER, W_SAME = 0.25, 0.15
