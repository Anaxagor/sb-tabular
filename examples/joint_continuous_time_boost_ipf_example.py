r"""
IPF-DSB with ONE time-conditioned CatBoost regressor per direction, on California Housing.

Registry id: ``dsb_ct_joint_gbt`` (IPF-DSB, continuous-time gamma grid, joint model, CatBoost MultiRMSE).

    python examples/joint_continuous_time_boost_ipf_example.py            # moderate CPU config, about 1.5 min on a laptop CPU
    python examples/joint_continuous_time_boost_ipf_example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset california_housing --output-root <root>
    python -m sbtab.experiments.tune --dataset california_housing --model dsb_ct_joint_gbt \
        --splits <root>/california_housing/splits.json --search-space configs/search_spaces/dsb_ct_joint_gbt.yaml
    python -m sbtab.experiments.cross_validate --dataset california_housing --model dsb_ct_joint_gbt \
        --selected-config <run>/tuning/selected_config.json --splits <root>/california_housing/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/california_housing/dsb_ct_joint_gbt/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "dsb_ct_joint_gbt"
DATASET = "california_housing"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
CONFIG = dict(
    # The boosted IPF solvers have no `horizon` key: T = sum(gamma). The legacy grid (gamma_max = 1e-2)
    # is far too short for the OU reference, so gamma_max is raised. The solver requires
    # alpha_ou * max(gamma) < 1 (here 0.1).
    gamma_min=1e-3, gamma_max=0.1, schedule="geom", num_steps=16, alpha_ou=1.0,
    ipf_iters=2,            # each half-iteration fits one regressor on (n_rows x num_steps) rows
    cb_iterations=100, cb_depth=5, cb_learning_rate=0.1, cb_thread_count=4,
    # heavier and better on this table (about 4 min): cb_iterations=200, cb_depth=6
)


def main(quick: bool = False):
    return run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)


if __name__ == "__main__":
    cli(main, __doc__)
