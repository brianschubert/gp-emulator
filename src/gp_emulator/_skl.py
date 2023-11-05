"""
Scikit-learn based GP emulators.
"""
import io
import pickle
import typing
from typing import Any
from typing_extensions import Self

import sklearn.gaussian_process as skl_gp
import sklearn.base as skl_base

from ._base import BaseGPEmulator, ExampleFeatureArray, MeanStd


class SklearnGPEmulator(BaseGPEmulator):
    """Emulator using Scikit-learn's GP implementation."""

    gaussian_process: skl_gp.GaussianProcessRegressor

    def __init__(
        self,
        gaussian_process: skl_gp.GaussianProcessRegressor,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.gaussian_process = typing.cast(
            skl_gp.GaussianProcessRegressor,
            skl_base.clone(gaussian_process),
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, file: io.RawIOBase) -> Self:
        return typing.cast(Self, pickle.load(file))

    def predict(self, x: ExampleFeatureArray) -> MeanStd[ExampleFeatureArray]:
        return MeanStd(*self.gaussian_process.predict(x, return_std=True))

    def export(self, file: io.RawIOBase) -> None:
        # Scikit-learn explicitly supports serializing models via pickling
        # https://scikit-learn.org/stable/model_persistence.html
        return pickle.dump(self, file)

    def train(self) -> None:
        self.gaussian_process.fit(self.train_features, self.train_targets)
