r"""
CSBM (categorical Schroedinger bridge matching, D-IMF) on a fully DISCRETE table: every column of
Car Evaluation has finite support. CSBM does not accept a schema with a continuous column.

Registry id: ``csbm`` (CSBM / D-IMF, finite states per column, uniform unit-horizon grid, torch MLP).

    python examples/csbm_example.py            # moderate CPU config, about 10 s on a laptop CPU
    python examples/csbm_example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset car_evaluation --output-root <root>
    python -m sbtab.experiments.tune --dataset car_evaluation --model csbm \
        --splits <root>/car_evaluation/splits.json --search-space configs/search_spaces/csbm.yaml
    python -m sbtab.experiments.cross_validate --dataset car_evaluation --model csbm \
        --selected-config <run>/tuning/selected_config.json --splits <root>/car_evaluation/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/car_evaluation/csbm/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "csbm"
DATASET = "car_evaluation"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
CONFIG = dict(
    num_outer_iterations=3,     # D-IMF outer iterations (forward + backward training in each)
    epochs=20,                  # training budget per direction and outer iteration
    num_steps=25,               # uniform grid on the unit horizon
    mixing_rate=1.0,            # fixed reference (the annealed variant is the separate id `csbm_annealed`)
    hidden_dim=128, lr=1e-3, batch_size=128, device="cpu",
)


def main(quick: bool = False):
    return run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)


if __name__ == "__main__":
    cli(main, __doc__)
