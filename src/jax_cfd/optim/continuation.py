"""
Penalisation continuation schedules for topology optimisation.  [Method 5]

Gradually increases alpha_max and decreases q (RAMP convexity parameter)
over the course of optimisation.  This prevents premature solidification
and improves convergence to binary (0/1) designs.

Typical schedule::

    schedule = make_continuation_schedule(
        alpha_stages=[500, 2000, 10000],
        q_stages=[0.1, 0.02, 0.005],
        steps_per_stage=30,
    )

    for stage in schedule:
        rho = optimize_stage(rho, stage.alpha_max, stage.q, stage.n_steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ContinuationState:
    """Parameters for one stage of penalisation continuation."""
    stage: int
    alpha_max: float
    q: float
    n_steps: int
    lr: float
    sobolev_eps: float  # Sobolev smoothing epsilon for this stage


def make_continuation_schedule(
    alpha_stages: List[float] = None,
    q_stages: List[float] = None,
    steps_per_stage: int = 30,
    lr_stages: List[float] = None,
    sobolev_stages: List[float] = None,
) -> List[ContinuationState]:
    """Build a multi-stage penalisation continuation schedule.

    Parameters
    ----------
    alpha_stages : list of float
        alpha_max values per stage (ascending).
        Default: [500, 2000, 10000]
    q_stages : list of float
        RAMP q values per stage (descending = sharper transition).
        Default: [0.1, 0.02, 0.005]
    steps_per_stage : int
        Optimisation iterations per stage.
    lr_stages : list of float, optional
        Learning rates per stage. Default: constant 0.01.
    sobolev_stages : list of float, optional
        Sobolev epsilon per stage. Default: decreases with stage.

    Returns
    -------
    schedule : list of ContinuationState
    """
    if alpha_stages is None:
        alpha_stages = [500.0, 2000.0, 10000.0]
    if q_stages is None:
        q_stages = [0.1, 0.02, 0.005]

    n_stages = len(alpha_stages)
    assert len(q_stages) == n_stages, "alpha_stages and q_stages must have same length"

    if lr_stages is None:
        lr_stages = [0.01] * n_stages
    if sobolev_stages is None:
        # Decrease smoothing as penalisation sharpens
        sobolev_stages = [4.0, 2.0, 1.0][:n_stages]
        while len(sobolev_stages) < n_stages:
            sobolev_stages.append(sobolev_stages[-1])

    if isinstance(steps_per_stage, int):
        steps_list = [steps_per_stage] * n_stages
    else:
        steps_list = list(steps_per_stage)

    schedule = []
    for i in range(n_stages):
        schedule.append(ContinuationState(
            stage=i,
            alpha_max=alpha_stages[i],
            q=q_stages[i],
            n_steps=steps_list[i],
            lr=lr_stages[i],
            sobolev_eps=sobolev_stages[i],
        ))

    return schedule


def default_schedule_from_alpha_max(
    alpha_max_final: float,
    n_stages: int = 3,
    steps_per_stage: int = 30,
) -> List[ContinuationState]:
    """Auto-generate a continuation schedule from a final alpha_max.

    Creates geometrically spaced alpha_max values and corresponding
    q values that decrease from 0.1 to 0.005.
    """
    import numpy as np

    # Geometric spacing for alpha_max
    alpha_min = max(alpha_max_final / (10 ** (n_stages - 1)), 100.0)
    alphas = np.geomspace(alpha_min, alpha_max_final, n_stages).tolist()

    # q decreases log-linearly
    qs = np.geomspace(0.1, 0.005, n_stages).tolist()

    return make_continuation_schedule(
        alpha_stages=alphas,
        q_stages=qs,
        steps_per_stage=steps_per_stage,
    )
