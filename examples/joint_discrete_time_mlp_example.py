r"""
IMF-DSBM with one MLP per edge and direction, on California Housing.

Registry id: ``dsbm_dt_joint_mlp`` (IMF-DSBM, discrete-time, joint model, torch MLP).

    python examples/joint_discrete_time_mlp_example.py            # moderate CPU config, about 45 s on a laptop CPU
    python examples/joint_discrete_time_mlp_example.py --quick    # a few seconds

``--quick`` uses the tiny bounded smoke config on <= 600 training rows. It only shows
that the code path runs: its output says NOTHING about model quality.

Data loading, the train-only preprocessing, fitting, sampling and scoring all go
through ``_common.run_example``, i.e. the dataset registry, ``CommonPreprocessor``,
the adapter contract and the canonical ``sbtab.evaluation`` metrics. The split is
illustrative. A result that is meant to be reported comes from the staged pipeline:

    python -m sbtab.experiments.prepare_splits --dataset california_housing --output-root <root>
    python -m sbtab.experiments.tune --dataset california_housing --model dsbm_dt_joint_mlp \
        --splits <root>/california_housing/splits.json --search-space configs/search_spaces/dsbm_dt_joint_mlp.yaml
    python -m sbtab.experiments.cross_validate --dataset california_housing --model dsbm_dt_joint_mlp \
        --selected-config <run>/tuning/selected_config.json --splits <root>/california_housing/splits.json
    python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json

with <run> = <root>/california_housing/dsbm_dt_joint_mlp/<run-id>; afterwards
``python -m sbtab.experiments.aggregate_results --output-root <root>`` tabulates every run.
"""
from _common import cli, run_example, smoke_config

MODEL_ID = "dsbm_dt_joint_mlp"
DATASET = "california_housing"

# Moderate config. Keys are the adapter's DEFAULTS (strict: an unknown key raises).
# The initial coupling is the independent coupling data (x) prior, which is the adapter default, and
# sampling always keeps the dynamics noise: `noise` is deliberately not an adapter option.
CONFIG = dict(
    n_stages=3,             # alternating IMF stages; the last one is the backward (generation) direction
    num_steps=16,           # uniform grid on the unit horizon -> 16 edge MLPs per direction
    sigma=0.3,              # diffusion of the Brownian reference
    n_epochs=10,            # training budget: epochs per edge MLP
    hidden_dim=128, n_layers=3, lr=1e-3, batch_size=256, device="cpu",
)


def main(quick: bool = False):
    return run_example(MODEL_ID, DATASET, smoke_config(MODEL_ID) if quick else CONFIG, quick=quick)


if __name__ == "__main__":
    cli(main, __doc__)
