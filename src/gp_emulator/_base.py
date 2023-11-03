import abc
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from nptyping import Floating, NDArray
import xarray as xr
from typing_extensions import TypeAlias

ExampleFeatureArray: TypeAlias = NDArray[Literal["* example, * feature"], Floating]
"""
2D array with examples along the first dimension and features along the second dimension. 
"""

CoordinateName: TypeAlias = str


class BaseGPEmulator(abc.ABC):
    """
    Base class for Gaussian Process (GP) based emulators.
    """

    dim_names: tuple[str, ...]
    """
    Fixed names of input parameters.
    
    All grid outputs will have these dimensions in this order.
    """

    def __init__(self) -> None:
        ...

    def predict_grid(self, x: Mapping[CoordinateName, Sequence[Any]]):
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

        return mean_grid, std_grid

    @abc.abstractmethod
    def predict(
        self, x: ExampleFeatureArray
    ) -> tuple[ExampleFeatureArray, ExampleFeatureArray]:
        ...
