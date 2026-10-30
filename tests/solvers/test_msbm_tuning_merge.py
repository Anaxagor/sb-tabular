"""Integration contracts for MSBM imported from feature/tuning."""
import pytest
import torch

from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver
from sbtab.solvers.msbm.reference import CategoricalReference


def config(**overrides):
    return MixedSBMConfig(**dict(dict(
        num_steps=4, fb_sequence=("b", "f", "b"), hidden_dim=8, n_layers=2,
        time_dim=4, cat_emb_dim=2, dropout=0.1, batch_size=8,
        steps_per_direction=3, min_steps_per_direction=0, seed=7,
    ), **overrides))


@pytest.mark.parametrize("n_rows", [3, 17, 65])
def test_fixed_budget_is_independent_of_dataset_size(n_rows):
    solver = MixedSBMSolver(1, [], [], config())
    solver.fit(torch.linspace(-1, 1, n_rows).reshape(-1, 1))
    assert [s["n_updates"] for s in solver.stage_log] == [3, 3, 3]
    assert solver.n_updates == 9


def test_epoch_budget_and_step_floor_are_consumed():
    solver = MixedSBMSolver(1, [], [], config(
        fb_sequence=("b",), steps_per_direction=None, epochs_per_direction=2,
        min_steps_per_direction=7,
    ))
    solver.fit(torch.arange(17, dtype=torch.float32).reshape(-1, 1))
    assert solver.n_updates == 7  # max(2 * ceil(17 / 8), 7)


def test_categorical_sampling_seed_is_independent_of_callers_and_survives_reload(tmp_path):
    solver = MixedSBMSolver(1, [3, 1], [False, False], config())
    x = torch.linspace(-1, 1, 17).reshape(-1, 1)
    cat = torch.stack((torch.arange(17) % 3, torch.zeros(17, dtype=torch.long)), dim=1)
    solver.fit(x, cat)
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    first = solver.sample(19, seed=42, batch_size=7)
    assert torch.equal(before, torch.get_rng_state())
    solver.save_checkpoint(tmp_path / "model.pt")
    reloaded = MixedSBMSolver.load_checkpoint(tmp_path / "model.pt")
    torch.manual_seed(999)
    second = reloaded.sample(19, seed=42, batch_size=7)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert reloaded.n_updates == solver.n_updates
    assert (first[1][:, 1] == 0).all()


def test_tuning_reference_is_separate_from_canonical_csbm_reference():
    from sbtab.bridge.reference import CategoricalReference as CanonicalReference
    from sbtab.bridge.timegrid import TimeGrid

    old = CategoricalReference([4], torch.tensor([False]), 8, alpha=0.2)
    new = CanonicalReference([4], torch.tensor([False]), TimeGrid.uniform(8))
    assert old.describe()["family"] == "feature_tuning_step_kernel"
    assert new.describe()["family"] == "generator_exponential"
    # Legacy alpha determines a per-step transition, independent of the grid size.
    other = CategoricalReference([4], torch.tensor([False]), 16, alpha=0.2)
    torch.testing.assert_close(old._powers[:, 1], other._powers[:, 1])
    assert not torch.allclose(old._powers[:, -1], other._powers[:, -1])


def test_csbm_backbone_keeps_default_checkpoint_keys_and_accepts_tuning_depth():
    from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP

    default = CSBMTableMLP([3, 5], emb_dim=2, hidden_dim=8, time_dim=4)
    assert default.state_dict()["net.4.weight"].shape == (10, 8)
    restored = CSBMTableMLP([3, 5], emb_dim=2, hidden_dim=8, time_dim=4)
    restored.load_state_dict(default.state_dict(), strict=True)
    deeper = CSBMTableMLP([3, 5], emb_dim=2, hidden_dim=8, time_dim=4,
                          n_layers=4, dropout=0.1)
    x, t = torch.tensor([[1, 4], [0, 2]]), torch.tensor([[0.25], [0.5]])
    assert deeper(x, t).shape == (2, 2, 5)
