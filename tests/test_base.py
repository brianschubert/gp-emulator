import warnings

# Temporarily silence deprecated alias warnings from nptyping 2.5.0 for numpy>=1.24.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping")

import io
from typing_extensions import Self

import numpy as np
import xarray as xr

import gp_emulator
from gp_emulator._base import ExampleFeatureArray


class DummyEmulator(gp_emulator.BaseGPEmulator):
    @classmethod
    def load(cls, file: io.RawIOBase) -> Self:
        raise NotImplementedError

    def export(self, file: io.RawIOBase) -> None:
        raise NotImplementedError

    def predict(
        self, x: ExampleFeatureArray
    ) -> tuple[ExampleFeatureArray, ExampleFeatureArray]:
        return x.sum(axis=1, keepdims=True), x.prod(axis=1, keepdims=True)

    def train(self) -> None:
        pass


def test_grid_shape() -> None:
    x = np.arange(4 * 2).reshape(4, 2)
    y = x.sum(axis=1, keepdims=True)
    emulator = DummyEmulator(x, y, ("a", "b"))

    test_grid = {"a": np.arange(10), "b": np.arange(5)}
    result = emulator.predict_grid(test_grid)

    assert result.mean.dims == tuple(test_grid.keys())
    assert result.std.dims == tuple(test_grid.keys())

    for k, v in test_grid.items():
        np.testing.assert_equal(result.mean[k], v)
        np.testing.assert_equal(result.std[k], v)

    np.testing.assert_equal(
        result.mean, test_grid["a"].reshape(-1, 1) + test_grid["b"].reshape(1, -1)
    )
    np.testing.assert_equal(
        result.std, test_grid["a"].reshape(-1, 1) * test_grid["b"].reshape(1, -1)
    )
