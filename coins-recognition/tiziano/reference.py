"""Builds the classification model from the reference_set images: the shared
shape atlas + colour prototypes (common/reference.py) plus this variant's SIFT
descriptor cache (sift_cue.py).

Same role as common/reference.py's `initialize_ght` stand-in, extended with
the SIFT half of the model.
"""
from pathlib import Path

from common.reference import build_reference_model as _build_base_model

from . import config as C
from .sift_cue import calibrate_sift_templates


def build_reference_model(reference_dir):
    """Builds {atlas, group_ab, proto, proto_sep, sift_cache} from the 8 reference images."""
    model = _build_base_model(reference_dir)

    reference_dir = Path(reference_dir)
    sift_templates = {name: str(reference_dir / f"{name}.jpg") for name in C.CLASSES}
    model["sift_cache"] = calibrate_sift_templates(sift_templates)

    return model
