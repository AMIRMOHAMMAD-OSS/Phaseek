import numpy as np

from phaseek_v3.fegs_fast import FastFEGSExtractor


def test_linear_dpc_has_no_wraparound():
    extractor = object.__new__(FastFEGSExtractor)
    extractor.legacy_wrap_dpc = False
    codes = np.array([0, 1], dtype=np.intp)
    _, dpc = extractor.composition_features(codes)
    assert dpc[0, 1] == 1.0
    assert dpc[1, 0] == 0.0
