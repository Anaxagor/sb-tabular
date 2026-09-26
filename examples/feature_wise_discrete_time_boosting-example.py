r"""
Feature-wise (structural) IMF-DSBM with CatBoost: a DAG is learned on the fit rows and every column
gets its own scalar bridge conditioned on its parents. California Housing.

Registry id: ``dsbm_dt_structural_gbt`` (IMF-DSBM, discrete-time, feature-wise: autoregressive chain by default, here a learned DAG; CatBoost RMSE).

    python examples/feature_wise_discrete_time_boosting-example.py            # moderate CPU config, about 30 s on a laptop CPU
    python examples/feature_wise_discrete_time_boosting-example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset california_housing --output-root <root>
    python -m sbtab.experiments.tune --dataset california_housing --model dsbm_dt_structural_gbt \
        --splits <root>/california_housing/splits.json --search-space configs/search_spaces/dsbm_dt_structural_gbt.yaml
    python -m sbtab.experiments.cross_validate --dataset california_housing --model dsbm_dt_structural_gbt \
        --selected-config <run>/tuning/selected_config.json --splits <root>/california_housing/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/california_housing/dsbm_dt_structural_gbt/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "dsbm_dt_structural_gbt"
DATASET = "california_housing"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
CONFIG = dict(
    # Feature-wise IMF-DSBM: for every column a scalar bridge conditioned on its parents, one CatBoost model
    # per column, edge and direction. structure="autoregressive" is the full chain in column order (an exact
    # factorisation); structure="learned" fits a DAG on the current fit rows instead.
    n_stages=3, num_steps=12, sigma=0.3,
    structure="learned", structure_n_bins=5,
    cb_iterations=60, cb_depth=4, cb_learning_rate=0.1, cb_thread_count=4,
)


def main(quick: bool = False):
    result = run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)
    structure = result["describe"]["structure"]     # kind "learned": fitted on the current fit rows only
    print(f"structure = {structure['kind']}; generation order and parents:")
    for column in structure["order"]:
        print(f"  {column} <- {list(structure['parents'].get(column, []))}")
    return result


if __name__ == "__main__":
    cli(main, __doc__)
