def run(config, **kwargs):
    """Lazy import to avoid circular loading when using ``python -m jax_cfd.runner``."""
    from jax_cfd.runner import run as _run
    return _run(config, **kwargs)

__all__ = ['run']
