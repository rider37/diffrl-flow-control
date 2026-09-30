"""Tests for HDF5 I/O (writer + reader round-trip)."""
import sys
import os
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState


@pytest.fixture
def tmp_h5(tmp_path):
    """Return a temporary HDF5 file path."""
    return str(tmp_path / 'test_run.h5')


@pytest.fixture
def grid_small():
    return MACGrid((8, 8), (1.0, 1.0))


@pytest.fixture
def state_small(grid_small):
    g = grid_small
    u = jnp.ones(g.face_shape(0)) * 0.5
    v = jnp.zeros(g.face_shape(1))
    p = jnp.ones(g.cell_shape()) * 0.1
    return FluidState(velocity=(u, v), pressure=p, scalars={}, ibm_forces=None)


class TestWriteRead:
    def test_write_and_read_snapshot(self, tmp_h5, grid_small, state_small):
        from jax_cfd.io.writer import H5Writer
        from jax_cfd.io.reader import load_snapshot, list_steps

        cfg = {'solver': 'incompressible_ns', 'Re': 100.0}

        with H5Writer(tmp_h5) as w:
            w.open(cfg, grid_small)
            w.write_snapshot(100, 1.0, state_small)
            w.write_snapshot(200, 2.0, state_small)

        steps = list_steps(tmp_h5)
        assert steps == [100, 200]

        snap = load_snapshot(tmp_h5, 100)
        assert abs(snap['t'] - 1.0) < 1e-5
        assert snap['step'] == 100
        assert 'u' in snap
        assert 'v' in snap
        assert 'p' in snap

    def test_read_grid(self, tmp_h5, grid_small, state_small):
        from jax_cfd.io.writer import H5Writer
        from jax_cfd.io.reader import load_grid

        cfg = {'solver': 'incompressible_ns', 'Re': 100.0}
        with H5Writer(tmp_h5) as w:
            w.open(cfg, grid_small)
            w.write_snapshot(100, 1.0, state_small)

        grid = load_grid(tmp_h5)
        assert 'xc' in grid
        assert 'yc' in grid
        assert len(grid['xc']) == 8
        assert len(grid['yc']) == 8

    def test_read_config(self, tmp_h5, grid_small, state_small):
        from jax_cfd.io.writer import H5Writer
        from jax_cfd.io.reader import load_config

        cfg = {'solver': 'incompressible_ns', 'Re': 100.0}
        with H5Writer(tmp_h5) as w:
            w.open(cfg, grid_small)

        loaded = load_config(tmp_h5)
        assert loaded['solver'] == 'incompressible_ns'
        assert loaded['Re'] == 100.0


class TestAppendMode:
    def test_append_continues(self, tmp_h5, grid_small, state_small):
        from jax_cfd.io.writer import H5Writer
        from jax_cfd.io.reader import list_steps

        cfg = {'solver': 'incompressible_ns', 'Re': 100.0}

        # Write initial
        with H5Writer(tmp_h5) as w:
            w.open(cfg, grid_small)
            w.write_snapshot(100, 1.0, state_small)

        # Append
        with H5Writer(tmp_h5) as w:
            w.open(cfg, grid_small, append=True)
            w.write_snapshot(200, 2.0, state_small)

        steps = list_steps(tmp_h5)
        assert steps == [100, 200]


class TestMonitor:
    def test_monitor_write_read(self, tmp_h5, grid_small, state_small):
        from jax_cfd.io.writer import H5Writer
        from jax_cfd.io.reader import load_monitor

        cfg = {'solver': 'incompressible_ns', 'Re': 100.0}
        with H5Writer(tmp_h5) as w:
            w.open(cfg, grid_small)
            w.append_monitor(100, 1.0, {'ke': 0.5, 'div': 1e-7})
            w.append_monitor(200, 2.0, {'ke': 0.6, 'div': 2e-7})

        mon = load_monitor(tmp_h5)
        assert len(mon['step']) == 2
        assert abs(mon['ke'][0] - 0.5) < 1e-5
        assert abs(mon['ke'][1] - 0.6) < 1e-5
