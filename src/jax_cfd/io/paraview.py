"""
Export JAX-CFD HDF5 output to XDMF + companion HDF5 for ParaView.

Supports both 2D and 3D simulations.

Workflow
--------
    from jax_cfd.io.paraview import export_xdmf
    xdmf = export_xdmf('output/run.h5')
    # → output/run_pv.xdmf   ← open this in ParaView
    # → output/run_pv.h5      ← cell-centred companion data
"""

from __future__ import annotations

import os
import numpy as np


def export_xdmf(
    h5_path: str,
    out_dir: str | None = None,
    steps: list[int] | None = None,
) -> str:
    """
    Convert a JAX-CFD HDF5 output file to XDMF + companion HDF5 for ParaView.

    Parameters
    ----------
    h5_path : path to the JAX-CFD output HDF5  (e.g. 'output/run.h5')
    out_dir : destination directory for the generated files;
              defaults to the same directory as *h5_path*
    steps   : integer step numbers to export;  ``None`` exports every snapshot

    Returns
    -------
    Absolute path to the generated .xdmf file.
    """
    import h5py

    h5_path = os.path.abspath(h5_path)
    if out_dir is None:
        out_dir = os.path.dirname(h5_path)
    os.makedirs(out_dir, exist_ok=True)

    stem          = os.path.splitext(os.path.basename(h5_path))[0]
    pv_h5_path    = os.path.join(out_dir, f'{stem}_pv.h5')
    pv_xdmf_path  = os.path.join(out_dir, f'{stem}_pv.xdmf')
    pv_h5_name    = os.path.basename(pv_h5_path)   # relative ref inside XDMF

    snap_meta = []  # (key, step, t, has_T)

    with h5py.File(h5_path, 'r') as src, h5py.File(pv_h5_path, 'w') as dst:

        # ── Detect dimensionality ────────────────────────────────────────
        ndim = int(src['grid'].attrs.get('ndim', 2))

        # ── Grid ─────────────────────────────────────────────────────────
        _axes  = ['x', 'y', 'z']
        _comps = ['u', 'v', 'w']
        face_keys = [f'{_axes[i]}{_comps[i]}' for i in range(ndim)]
        cc_keys   = [f'{_axes[i]}c' for i in range(ndim)]

        face_coords = [src['grid'][k][:] for k in face_keys]  # (Ni+1,) each
        cc_coords   = [src['grid'][k][:] for k in cc_keys]    # (Ni,) each
        shape_cc    = tuple(len(c) for c in cc_coords)         # (Nx, Ny[, Nz])
        spacings    = [float(c[1] - c[0]) if len(c) > 1 else float(face_coords[i][1] - face_coords[i][0])
                       for i, c in enumerate(cc_coords)]

        grd = dst.create_group('grid')
        for k, arr in zip(face_keys, face_coords):
            grd.create_dataset(k, data=arr)

        # ── Snapshots ────────────────────────────────────────────────────
        all_keys = sorted(src['fields'].keys())
        if steps is not None:
            step_set = {int(s) for s in steps}
            all_keys = [k for k in all_keys
                        if int(k.split('_')[-1]) in step_set]

        n_total = len(all_keys)
        for idx, key in enumerate(all_keys, 1):
            grp  = src['fields'][key]
            step = int(grp.attrs['step'])
            t    = float(grp.attrs['t'])

            # Load and interpolate velocity to cell centres
            vel_cc = []
            for i in range(ndim):
                u_mac = grp[_comps[i]][:]
                sl_p = [slice(None)] * ndim
                sl_m = [slice(None)] * ndim
                sl_p[i] = slice(1, None)
                sl_m[i] = slice(None, -1)
                vel_cc.append(0.5 * (u_mac[tuple(sl_p)] + u_mac[tuple(sl_m)]))

            p = grp['p'][:]
            speed = np.sqrt(sum(v**2 for v in vel_cc))

            has_T = 'T' in grp
            T_arr = grp['T'][:] if has_T else None
            has_phi = 'phi' in grp
            phi_arr = grp['phi'][:] if has_phi else None

            # Transpose for XDMF row-major: (Nx, Ny) -> (Ny, Nx),
            #                                (Nx, Ny, Nz) -> (Nz, Ny, Nx)
            rev_axes = tuple(range(ndim - 1, -1, -1))

            sg = dst.create_group(key)
            sg.attrs['step'] = step
            sg.attrs['t']    = t
            _cmp = dict(compression='gzip', compression_opts=4)

            for i in range(ndim):
                sg.create_dataset(_comps[i], data=vel_cc[i].transpose(rev_axes), **_cmp)
            sg.create_dataset('p',     data=p.transpose(rev_axes),     **_cmp)
            sg.create_dataset('speed', data=speed.transpose(rev_axes), **_cmp)

            if ndim == 2:
                vort = (np.gradient(vel_cc[1], spacings[0], axis=0)
                        - np.gradient(vel_cc[0], spacings[1], axis=1))
                sg.create_dataset('vorticity', data=vort.transpose(rev_axes), **_cmp)
            else:
                # 3D: vorticity magnitude
                from jax_cfd.core.operators import vorticity as _vort_fn
                import jax.numpy as jnp
                from jax_cfd.core.mac_grid import MACGrid
                _grid = MACGrid(shape_cc, tuple(float(fc[-1]) for fc in face_coords))
                _vel = tuple(jnp.array(grp[_comps[i]][:]) for i in range(ndim))
                ox, oy, oz = _vort_fn(_vel, _grid)
                vort_mag = np.array(jnp.sqrt(ox**2 + oy**2 + oz**2))
                sg.create_dataset('vorticity_mag', data=vort_mag.transpose(rev_axes), **_cmp)

            # Velocity vector (always 3-component for ParaView)
            rev_shape = tuple(shape_cc[i] for i in rev_axes)
            vel_vec = np.zeros(rev_shape + (3,), dtype=np.float64)
            for i in range(ndim):
                vel_vec[..., i] = vel_cc[i].transpose(rev_axes)
            sg.create_dataset('velocity', data=vel_vec, **_cmp)

            if has_T:
                sg.create_dataset('T', data=T_arr.transpose(rev_axes), **_cmp)
            if has_phi:
                sg.create_dataset('phi', data=phi_arr.transpose(rev_axes), **_cmp)

            snap_meta.append((key, step, t, has_T, has_phi))
            print(f'\r[paraview] {idx}/{n_total}  step={step}  t={t:.3f}',
                  end='', flush=True)

    print()  # newline after progress line

    # ── Build XDMF XML ────────────────────────────────────────────────────
    shape_n = tuple(len(fc) for fc in face_coords)  # node counts (Nx+1, ...)
    rev_shape_cc = tuple(shape_cc[i] for i in range(ndim - 1, -1, -1))
    rev_shape_n  = tuple(shape_n[i]  for i in range(ndim - 1, -1, -1))
    dims_cc_str = ' '.join(str(s) for s in rev_shape_cc)
    dims_n_str  = ' '.join(str(s) for s in rev_shape_n)

    if ndim == 2:
        topo_type = '2DRectMesh'
        geom_type = 'VXVY'
    else:
        topo_type = '3DRectMesh'
        geom_type = 'VXVYVZ'

    def _scalar(name, key):
        return [
            f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">',
            f'          <DataItem Dimensions="{dims_cc_str}" NumberType="Float"'
            f' Precision="8" Format="HDF">',
            f'            {pv_h5_name}:/{key}/{name}',
            f'          </DataItem>',
            f'        </Attribute>',
        ]

    lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="2.0">',
        '  <Domain>',
        '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">',
    ]

    for key, step, t, has_T, has_phi in snap_meta:
        lines += [
            f'      <Grid Name="{key}" GridType="Uniform">',
            f'        <Time Value="{t:.6f}"/>',
            f'        <Topology TopologyType="{topo_type}"'
            f' NumberOfElements="{dims_n_str}"/>',
            f'        <Geometry GeometryType="{geom_type}">',
        ]
        for i in range(ndim):
            lines += [
                f'          <DataItem Dimensions="{shape_n[i]}" NumberType="Float"'
                f' Precision="8" Format="HDF">',
                f'            {pv_h5_name}:/grid/{face_keys[i]}',
                f'          </DataItem>',
            ]
        lines += [f'        </Geometry>']

        scalar_names = list(_comps[:ndim]) + ['p', 'speed']
        if ndim == 2:
            scalar_names.append('vorticity')
        else:
            scalar_names.append('vorticity_mag')

        for fname in scalar_names:
            lines += _scalar(fname, key)

        if has_T:
            lines += _scalar('T', key)
        if has_phi:
            lines += _scalar('phi', key)

        # Velocity vector
        vel_dims = f'{dims_cc_str} 3'
        lines += [
            f'        <Attribute Name="velocity" AttributeType="Vector" Center="Cell">',
            f'          <DataItem Dimensions="{vel_dims}" NumberType="Float"'
            f' Precision="8" Format="HDF">',
            f'            {pv_h5_name}:/{key}/velocity',
            f'          </DataItem>',
            f'        </Attribute>',
            f'      </Grid>',
        ]

    lines += [
        '    </Grid>',
        '  </Domain>',
        '</Xdmf>',
    ]

    with open(pv_xdmf_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    if snap_meta:
        t0 = snap_meta[0][2]
        t1 = snap_meta[-1][2]
        print(f'[paraview] Exported {len(snap_meta)} snapshots'
              f'  t=[{t0:.3f}, {t1:.3f}]')
        print(f'[paraview] Open in ParaView: {pv_xdmf_path}')

    return pv_xdmf_path
