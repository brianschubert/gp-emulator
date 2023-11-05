from __future__ import annotations

import abc
import io
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Generic, Iterable, Literal, NamedTuple, TypeVar

import numpy as np
import xarray
from nptyping import Floating, NDArray
import xarray as xr
from typing_extensions import Self, TypeAlias

ExampleFeatureArray: TypeAlias = NDArray[Literal["* example, * feature"], Floating]
"""
2D array with examples along the first dimension and features along the second dimension. 
"""

_DimName: TypeAlias = str

_T = TypeVar("_T")

_G = TypeVar("_G", bound="BaseGPEmulator")


class MeanStd(NamedTuple, Generic[_T]):
    """Tuple containing a mean and standard deviation."""

    mean: _T
    std: _T


class BaseGPEmulator(abc.ABC):
    """
    Base class for Gaussian Process (GP) emulators.
    """

    dim_names: tuple[_DimName, ...]
    """
    Fixed names of input parameters.
    
    All grid outputs will have these dimensions in this order.
    """

    train_features: ExampleFeatureArray

    train_targets: ExampleFeatureArray

    def __init__(
        self,
        train_features: ExampleFeatureArray,
        train_targets: train_targets,
        dim_names: Iterable[_DimName],
    ) -> None:
        self.train_features = train_features
        self.train_targets = train_targets
        self.dim_names = tuple(dim_names)

        x_shape = np.shape(train_features)
        y_shape = np.shape(train_targets)

        if not len(x_shape) == len(y_shape) == 2:
            raise ValueError(
                f"training features and targets must be 2D arrays, got shapes {x_shape} and {y_shape}"
            )

        if x_shape[0] != y_shape[0]:
            raise ValueError(
                "training features must contain the same number of rows as the the targets"
            )

        if y_shape[1] != 1:
            raise ValueError("targets must have a single column")

    @classmethod
    def extract_dataarray(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, file: io.RawIOBase) -> Self:
        ...

    @abc.abstractmethod
    def train(self) -> None:
        """Fit this emulator to its training dataset."""

    def predict_grid(
        self, x: Mapping[_DimName, Sequence[Any]]
    ) -> MeanStd[xarray.DataArray]:
        expected_dims = set(self.dim_names)
        received_dims = set(x.keys())

        missing = expected_dims - received_dims
        extra = received_dims - expected_dims
        if missing or extra:
            raise ValueError(
                f"bad grid dimensions - expected {self.dim_names}, "
                f"got {received_dims}"
                f"(missing: {missing}, extraneous: {extra})"
            )

        dim_values = [x[d] for d in self.dim_names]

        # Note: creates dense copies. This is more performant at the cost of higher
        # memory use. If memory use becomes a problem, this can be updated to use
        # copy=False, which uses views.
        input_meshes = np.meshgrid(*dim_values, indexing="ij")
        mesh_shape = input_meshes[0].shape

        input_examples = np.hstack([mesh.reshape(-1, 1) for mesh in input_meshes])
        del input_meshes

        mean_examples, std_examples = self.predict(input_examples)

        mean_grid = xr.DataArray(
            mean_examples.reshape(mesh_shape), coords=x, dims=self.dim_names
        )
        std_grid = xr.DataArray(
            std_examples.reshape(mesh_shape), coords=x, dims=self.dim_names
        )

        return MeanStd(mean_grid, std_grid)

    @abc.abstractmethod
    def predict(self, x: ExampleFeatureArray) -> MeanStd[ExampleFeatureArray]:
        ...

    @abc.abstractmethod
    def export(self, file: io.RawIOBase) -> None:
        ...


class MultiGPEmulator(Generic[_G]):
    """
    Collection of GPs that can be evaluated simultaneously.
    """

    component_dim: _DimName

    component_names: list[Any]

    components: list[_G]

    component_factory: Callable[[], _G]

    def __init__(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def predict_components_grid(
        self,
        x: Mapping[_DimName, Sequence[Any]],
        which: Iterable[int] | None = None,
        format: Literal["array", "dataset", "tuple"] = "array",
    ) -> MeanStd[xr.DataArray]:
        ...

    @abc.abstractmethod
    def predict_components(
        self, x: ExampleFeatureArray, which: Iterable[int] | None = None
    ) -> Iterator[MeanStd]:
        if which is not None:
            desired_components = [self.components[idx] for idx in which]
        else:
            desired_components = self.components

        return (comp.predict(x) for comp in desired_components)
