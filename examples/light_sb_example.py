r"""
LightSB (Gaussian-mixture potential, diagonal covariance, exact conditional sampler), on California
Housing. This is LightSB, NOT LightSB-M.

Registry id: ``lightsb`` (LightSB, static potential, joint model, torch).

    python examples/light_sb_example.py            # moderate CPU config, about 10 s on a laptop CPU
    python examples/light_sb_example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset california_housing --output-root <root>
    python -m sbtab.experiments.tune --dataset california_housing --model lightsb \
        --splits <root>/california_housing/splits.json --search-space configs/search_spaces/lightsb.yaml
    python -m sbtab.experiments.cross_validate --dataset california_housing --model lightsb \
        --selected-config <run>/tuning/selected_config.json --splits <root>/california_housing/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/california_housing/lightsb/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "lightsb"
DATASET = "california_housing"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
CONFIG = dict(
    n_potentials=32,        # mixture components of the potential
    epsilon=0.1,            # entropic regularisation (variance of the Brownian reference)
    max_iter=5000,          # training budget: optimiser updates
    lr=1e-2, batch_size=256, device="cpu",
)


def main(quick: bool = False):
    return run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)


if __name__ == "__main__":
    cli(main, __doc__)
