r"""
IMF-DSBM with ONE time-conditioned CatBoost regressor per direction, on California Housing.

Registry id: ``dsbm_ct_joint_gbt`` (IMF-DSBM, unit-horizon Brownian reference, time-conditioned joint drift, CatBoost MultiRMSE).

    python examples/joint_continuous_time_boost_example.py            # moderate CPU config, about 30 s on a laptop CPU
    python examples/joint_continuous_time_boost_example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset california_housing --output-root <root>
    python -m sbtab.experiments.tune --dataset california_housing --model dsbm_ct_joint_gbt \
        --splits <root>/california_housing/splits.json --search-space configs/search_spaces/dsbm_ct_joint_gbt.yaml
    python -m sbtab.experiments.cross_validate --dataset california_housing --model dsbm_ct_joint_gbt \
        --selected-config <run>/tuning/selected_config.json --splits <root>/california_housing/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/california_housing/dsbm_ct_joint_gbt/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "dsbm_ct_joint_gbt"
DATASET = "california_housing"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
CONFIG = dict(
    # IMF-DSBM on the unit horizon: one time-conditioned CatBoost regressor per direction, refit from scratch
    # at every IMF stage on (n_rows x n_noise_per_pair) bridge samples.
    n_stages=3,                 # alternating stages; the last one is the backward (generation) direction
    num_steps=50, sigma=0.3,    # Euler-Maruyama steps of the sampler; Brownian reference diffusion
    n_noise_per_pair=4,         # (t, eps) draws per coupled pair: a time-conditioned regressor needs several
    cb_iterations=150, cb_depth=5, cb_learning_rate=0.1, cb_thread_count=4,
)


def main(quick: bool = False):
    return run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)


if __name__ == "__main__":
    cli(main, __doc__)
