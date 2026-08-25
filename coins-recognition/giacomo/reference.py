"""giacomo's reference model is exactly the shared base builder -- see
common/reference.py. Re-exported here so giacomo/pipeline.py's import stays
local to the package, matching the initialize_ght contract described there.
"""
from common.reference import build_reference_model  # noqa: F401
