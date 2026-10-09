r"""
IMF-DSBM with CatBoost drifts (one MultiRMSE model per edge and direction), on California Housing.

Registry id: ``dsbm_dt_joint_gbt`` (IMF-DSBM, discrete-time, joint model, CatBoost MultiRMSE).

    python examples/boosted_dsbm_example.py            # moderate CPU config, about 1.5 min on a laptop CPU
    python examples/boosted_dsbm_example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset california_housing --output-root <root>
    python -m sbtab.experiments.tune --dataset california_housing --model dsbm_dt_joint_gbt \
        --splits <root>/california_housing/splits.json --search-space configs/search_spaces/dsbm_dt_joint_gbt.yaml
    python -m sbtab.experiments.cross_validate --dataset california_housing --model dsbm_dt_joint_gbt \
        --selected-config <run>/tuning/selected_config.json --splits <root>/california_housing/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/california_housing/dsbm_dt_joint_gbt/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "dsbm_dt_joint_gbt"
DATASET = "california_housing"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
# The initial coupling is the independent coupling data (x) prior, which is the adapter default, and
# sampling always keeps the dynamics noise: `noise` is deliberately not an adapter option.
CONFIG = dict(
    n_stages=3,             # alternating IMF stages; the last one is the backward (generation) direction
    num_steps=12,           # uniform grid on the unit horizon -> 12 edge models per direction
    sigma=0.3,              # diffusion of the Brownian reference
    cb_iterations=100, cb_depth=5, cb_learning_rate=0.1, cb_thread_count=4,
    # heavier and better on this table (about 6 min): num_steps=16, cb_iterations=200, cb_depth=6
)


def main(quick: bool = False):
    return run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)


if __name__ == "__main__":
    cli(main, __doc__)
