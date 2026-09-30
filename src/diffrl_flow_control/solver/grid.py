"""Uniform staggered (Harlow--Welch MAC) grid.

Pressure and scalars are stored at the cell centres, shape (N_1, ..., N_D); the
i-th velocity component is stored on the i-faces, whose i-th dimension has
N_i + 1 entries.
"""
from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class MACGrid:
    shape: Tuple[int, ...]          # (Nx, Ny)
    domain_size: Tuple[float, ...]  # (Lx, Ly)

    def __post_init__(self):
        assert len(self.shape) == len(self.domain_size), \
            "Shape and size must be same dimensions."

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spacings(self) -> Tuple[float, ...]:
        """Cell sizes L/N per axis."""
        return tuple(L / N for L, N in zip(self.domain_size, self.shape))

    def spacings_array(self, axis: int) -> float:
        """Cell size along ``axis``."""
        return self.spacings[axis]

    def face_coords_1d(self, axis: int) -> jnp.ndarray:
        """Face positions along ``axis`` (length N+1)."""
        L = self.domain_size[axis]
        N = self.shape[axis]
        return jnp.linspace(0.0, L, N + 1)

    def cell_centers_1d(self, axis: int) -> jnp.ndarray:
        """Cell-centre positions along ``axis`` (length N)."""
        fc = self.face_coords_1d(axis)
        return 0.5 * (fc[:-1] + fc[1:])

    def cell_shape(self) -> Tuple[int, ...]:
        return self.shape

    def face_shape(self, axis: int) -> Tuple[int, ...]:
        """Array shape of the velocity component normal to ``axis``."""
        face_shp = list(self.shape)
        face_shp[axis] += 1
        return tuple(face_shp)

    def cell_centers(self) -> Tuple[jnp.ndarray, ...]:
        """1-D cell-centre coordinates along each axis."""
        return tuple(self.cell_centers_1d(ax) for ax in range(self.ndim))

    def face_centers(self, axis: int) -> Tuple[jnp.ndarray, ...]:
        """1-D coordinates of the ``axis``-faces along each axis."""
        coords = list(self.cell_centers())
        coords[axis] = self.face_coords_1d(axis)
        return tuple(coords)
