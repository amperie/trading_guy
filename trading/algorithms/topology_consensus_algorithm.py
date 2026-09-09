from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "promoted"
    / "Topology-Based_UPRO_Predictor_Dual_Feature_B0_B1_Consensus_backtest_2a924236"
    / "TopologyConsensusAlgorithm.py"
)
_SPEC = spec_from_file_location("_promoted_topology_consensus_algorithm", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load promoted topology algorithm from {_SOURCE}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

TopologyConsensusAlgorithm = _MODULE.TopologyConsensusAlgorithm
compute_persistent_homology_b0_b1 = _MODULE.compute_persistent_homology_b0_b1

__all__ = ["TopologyConsensusAlgorithm", "compute_persistent_homology_b0_b1"]
