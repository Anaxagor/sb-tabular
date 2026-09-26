r"""
MixedSBM on its NATIVE mixed state space: continuous columns in a Brownian block, discrete and
categorical columns as finite states. Insurance (3 continuous, 1 discrete, 3 categorical columns).

Registry id: ``mixedsbm`` (MixedSBM, R^d x finite states, one uniform unit-horizon grid, torch MLP).

    python examples/mixedsbm_example.py            # moderate CPU config, about 15 s on a laptop CPU
    python examples/mixedsbm_example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset insurance --output-root <root>
    python -m sbtab.experiments.tune --dataset insurance --model mixedsbm \
        --splits <root>/insurance/splits.json --search-space configs/search_spaces/mixedsbm.yaml
    python -m sbtab.experiments.cross_validate --dataset insurance --model mixedsbm \
        --selected-config <run>/tuning/selected_config.json --splits <root>/insurance/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/insurance/mixedsbm/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "mixedsbm"
DATASET = "insurance"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
CONFIG = dict(
    n_stages=3,                 # alternating stages; the last one is the backward (generation) direction
    epochs_per_direction=100,   # training budget per stage (the table has only ~1k rows)
    num_steps=50, sigma=0.3,    # numerical block: Brownian reference on the unit horizon
    cat_mixing_rate=1.0,        # finite-state block: mixing rate of the reference
    hidden_dim=256, n_layers=3, dropout=0.0, lr=1e-3, batch_size=128, device="cpu",
)


def main(quick: bool = False):
    return run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)


if __name__ == "__main__":
    cli(main, __doc__)
