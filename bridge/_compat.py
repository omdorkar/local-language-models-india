"""Import shims so IndicXlit runs without TensorFlow.

`ai4bharat-transliteration` does `from urduhack import normalize` at import time, but
`urduhack/__init__` pulls in TensorFlow (for its NER/POS models). The Roman->Indic direction
never calls that function (it is only used when the *source* is an Indic script), and
`urduhack.normalization` itself is pure Python. So we register a bare `urduhack` package
(pointing at the real installed directory, without executing its __init__) so that the
real `urduhack.normalization.normalize` is imported and TensorFlow is never needed.
"""
import importlib.util
import sys
import types


def patch_urduhack() -> None:
    if "urduhack" in sys.modules:
        return
    spec = importlib.util.find_spec("urduhack")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("urduhack is not installed (pip install --no-deps urduhack==1.1.1)")
    pkg = types.ModuleType("urduhack")
    pkg.__path__ = list(spec.submodule_search_locations)
    pkg.__file__ = spec.origin
    sys.modules["urduhack"] = pkg
    from urduhack.normalization import normalize  # real implementation, pure Python

    pkg.normalize = normalize
