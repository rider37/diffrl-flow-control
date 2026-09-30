"""Tests for config validation."""
import pytest

from jax_cfd.runner import _validate


def _make_valid_cfg(**overrides):
    """Return a minimal valid config dict, with optional overrides."""
    cfg = {
        'solver': 'incompressible_ns',
        'domain': {'cells': [16, 16], 'size': [1.0, 1.0]},
        'Re': 100.0,
        'boundaries': {
            'x_min': {'type': 'wall_no_slip'},
            'x_max': {'type': 'wall_no_slip'},
            'y_min': {'type': 'wall_no_slip'},
            'y_max': {'type': 'wall_no_slip'},
        },
    }
    cfg.update(overrides)
    return cfg


class TestSolverValidation:
    def test_valid_config_passes(self):
        _validate(_make_valid_cfg())

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            _validate(_make_valid_cfg(solver='magic_solver'))

    def test_missing_Re_raises(self):
        cfg = _make_valid_cfg()
        del cfg['Re']
        with pytest.raises(ValueError, match="requires config keys"):
            _validate(cfg)


class TestDomainValidation:
    def test_missing_domain_raises(self):
        cfg = _make_valid_cfg()
        del cfg['domain']
        with pytest.raises(ValueError, match="'domain' dict"):
            _validate(cfg)

    def test_missing_cells_raises(self):
        cfg = _make_valid_cfg()
        del cfg['domain']['cells']
        with pytest.raises(ValueError, match="domain.cells"):
            _validate(cfg)

    def test_missing_size_raises(self):
        cfg = _make_valid_cfg()
        del cfg['domain']['size']
        with pytest.raises(ValueError, match="domain.size"):
            _validate(cfg)

    def test_mismatched_dims_raises(self):
        cfg = _make_valid_cfg()
        cfg['domain']['cells'] = [16, 16]
        cfg['domain']['size'] = [1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="length"):
            _validate(cfg)

    def test_zero_cells_raises(self):
        cfg = _make_valid_cfg()
        cfg['domain']['cells'] = [0, 16]
        with pytest.raises(ValueError, match="positive integer"):
            _validate(cfg)

    def test_negative_size_raises(self):
        cfg = _make_valid_cfg()
        cfg['domain']['size'] = [-1.0, 1.0]
        with pytest.raises(ValueError, match="positive"):
            _validate(cfg)

    def test_4d_raises(self):
        cfg = _make_valid_cfg()
        cfg['domain']['cells'] = [8, 8, 8, 8]
        cfg['domain']['size'] = [1.0, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="2D and 3D"):
            _validate(cfg)


class TestPhysicsValidation:
    def test_negative_Re_raises(self):
        with pytest.raises(ValueError, match="Re="):
            _validate(_make_valid_cfg(Re=-100))

    def test_negative_Pr_raises(self):
        cfg = _make_valid_cfg(solver='incompressible_ns_heat', Pr=-0.5)
        with pytest.raises(ValueError, match="Pr="):
            _validate(cfg)

    def test_negative_Ra_raises(self):
        with pytest.raises(ValueError, match="Ra="):
            _validate(_make_valid_cfg(Ra=-1e6))


class TestTimeControlValidation:
    def test_negative_cfl_raises(self):
        with pytest.raises(ValueError, match="max_cfl"):
            _validate(_make_valid_cfg(time_control={'max_cfl': -0.3}))

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end"):
            _validate(_make_valid_cfg(time_control={'t_end': -10.0}))

    def test_zero_diag_every_raises(self):
        with pytest.raises(ValueError, match="diag_every"):
            _validate(_make_valid_cfg(time_control={'diag_every': 0}))


class TestBCValidation:
    def test_unknown_bc_type_raises(self):
        cfg = _make_valid_cfg()
        cfg['boundaries']['x_min'] = {'type': 'magic_boundary'}
        with pytest.raises(ValueError, match="Unknown BC type"):
            _validate(cfg)

    def test_unpaired_periodic_raises(self):
        cfg = _make_valid_cfg()
        cfg['boundaries']['x_min'] = {'type': 'periodic'}
        cfg['boundaries']['x_max'] = {'type': 'wall_no_slip'}
        with pytest.raises(ValueError, match="Periodic BC"):
            _validate(cfg)

    def test_paired_periodic_passes(self):
        cfg = _make_valid_cfg()
        cfg['boundaries']['x_min'] = {'type': 'periodic'}
        cfg['boundaries']['x_max'] = {'type': 'periodic'}
        _validate(cfg)  # should not raise


class TestSchemeValidation:
    def test_unknown_time_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown time scheme"):
            _validate(_make_valid_cfg(schemes={'time': 'rk99'}))

    def test_unknown_advection_raises(self):
        with pytest.raises(ValueError, match="Unknown advection"):
            _validate(_make_valid_cfg(schemes={'advection': 'superbee99'}))

    def test_unknown_diffusion_raises(self):
        with pytest.raises(ValueError, match="Unknown diffusion"):
            _validate(_make_valid_cfg(schemes={'diffusion': 'magic'}))


class TestTurbulenceValidation:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown turbulence model"):
            _validate(_make_valid_cfg(turbulence={'model': 'k_epsilon'}))

    def test_valid_models_pass(self):
        for m in ['laminar', 'smagorinsky', 'wale', 'dynamic_smagorinsky']:
            _validate(_make_valid_cfg(turbulence={'model': m}))
