from .base import BaseDatasetBuilder
from .greenbyte import GreenbyteDatasetBuilder
from .hill_of_towie import HillOfTowieDatasetBuilder
from .sdwpf_kddcup import SDWPFKDDCupDatasetBuilder

_BUILDERS = {
    "greenbyte": GreenbyteDatasetBuilder,
    "hill_of_towie": HillOfTowieDatasetBuilder,
    "sdwpf_kddcup": SDWPFKDDCupDatasetBuilder,
}


def get_builder(spec, cache_root):
    builder_cls = _BUILDERS[spec.handler]
    return builder_cls(spec=spec, cache_root=cache_root)


__all__ = ["BaseDatasetBuilder", "get_builder"]
