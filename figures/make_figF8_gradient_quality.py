"""Fig. F.1: cosine similarity of AD and FD gradients.

Rendered with make_fig7 of figures/pof_make_figures.py from results/gradient_quality/.
Writes figures/figF8_gradient_quality.pdf.
"""
import os, sys, inspect, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
spec = importlib.util.spec_from_file_location('pof_make_figures', os.path.join(HERE, 'pof_make_figures.py'))
mf = importlib.util.module_from_spec(spec); sys.modules['pof_make_figures'] = mf; spec.loader.exec_module(mf)
src = inspect.getsource(mf.make_fig7)
src = src.replace("cosine similarity (AD vs FD)", r"$\\cos(\\boldsymbol{g}_{\\mathrm{AD}},\\boldsymbol{g}_{\\mathrm{FD}})$")
src = src.replace("ax.set_ylabel('cosine similarity')", "ax.set_ylabel(r'$\\cos(\\boldsymbol{g}_{\\mathrm{AD}},\\boldsymbol{g}_{\\mathrm{FD}})$')")
src = src.replace("ax.set_xlabel('Ra')", "ax.set_xlabel(r'$\\mathrm{Ra}$')")
src = src.replace("ax.set_xlabel(r'rollout horizon $h$')", "ax.set_xlabel(r'horizon $h$')")
src = src.replace("markersize=5, markerfacecolor='none', markeredgewidth=1.0,", "markersize=6, markerfacecolor='none', markeredgewidth=1.2,")
src = src.replace("label=r'mean $\\pm$ std'", "label=r'Mean $\\pm$ SD'").replace("label='median'", "label='Median'")
src = src.replace("ax.set_xlabel(r'horizon $h$')", "ax.set_xlabel(r'Horizon $h$')")
src = src.replace("fig7_gradient_quality", "figF8_gradient_quality")
src = src.replace("figsize=(W_DOUBLE, 2.8)", "figsize=(5.4, 2.7)")
src = src.replace("ax.legend(loc='lower right', fontsize=8)", "ax.legend(loc='lower right', fontsize=9.5)")

src = src.replace("    panel_label(ax, 'a')\n", """    panel_label(ax, 'a')
    # inset: zoom on Ra=1e5 and 1e6, whose per-state values crowd near 1
    axin = ax.inset_axes([0.36, 0.30, 0.60, 0.46])
    for i, d in enumerate(cos_data):
        if d['Ra'] < 5e4:
            continue
        per_state = [r['cosine_similarity'] for r in d['results']]
        jitter = rng.uniform(-0.10, 0.10, size=len(per_state))
        axin.scatter(x[i] + jitter, per_state, s=12, facecolor='white',
                     edgecolor=C_SHAC, linewidth=0.7, alpha=0.85, zorder=2)
        axin.errorbar([x[i]], [cos_means[i]], yerr=[cos_stds[i]], fmt='o',
                      color=C_SHAC, markersize=5, capsize=3, elinewidth=1.2,
                      capthick=1.2, zorder=4)
        axin.plot([x[i]], [cos_medians[i]], 's', color=C_PPO, markersize=5,
                  markerfacecolor='none', markeredgewidth=1.2, zorder=5)
    axin.axhline(0.95, color=C_GRAY, ls='--', alpha=0.5, lw=0.6)
    axin.set_xlim(0.6, 2.4); axin.set_ylim(0.88, 1.005)
    axin.set_xticks([1, 2]); axin.set_xticklabels([r'$10^5$', r'$10^6$'])
    axin.set_yticks([0.90, 0.95, 1.00])
    axin.tick_params(labelsize=9, length=2)
    axin.set_title('Zoom', fontsize=9, pad=2)
""")
mf.plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
                        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                        'legend.fontsize': 9.5})
src = src.replace("panel_label(ax, 'a')", "panel_label(ax, 'a', fontsize=11)").replace("panel_label(ax, 'b')", "panel_label(ax, 'b', fontsize=11)")
src = src.replace("panel_label(ax, 'a', fontsize=11)", "ax.text(0.0, 1.02, '(a)', transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold')").replace("panel_label(ax, 'b', fontsize=11)", "ax.text(0.0, 1.02, '(b)', transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold')")
exec(src, mf.__dict__)
mf.FIG_DIR = HERE
mf.make_fig7()
print('wrote', os.path.join(HERE, 'figF8_gradient_quality.pdf'))
