"""Default matrix: one basic joint MLP per DSB family; explicit variants remain available."""
from sbtab.solvers.registry import get_entry, solver_registry
from sbtab.experiments.experiment_common import StageError


BASIC_DSB_MODELS = {
    "IPF-DSB": "dsb_ct_joint_mlp",
    "IMF-DSBM": "dsbm_ct_joint_mlp",
}


def exclusion_reason(model_id: str):
    entry = get_entry(model_id)
    selected = BASIC_DSB_MODELS.get(entry.family)
    if selected is not None and model_id != selected:
        return f"excluded from default experiments: {entry.family} defaults to the basic joint MLP {selected!r}; select this variant explicitly"
    return None


def require_experiment_model(model_id: str):
    entry = get_entry(model_id)
    if entry.status == "unavailable":
        raise StageError("undefined", f"model {model_id!r} is unavailable for experiments: {entry.notes}")
    return entry


def unavailable_experiment_reason(model_id: str):
    """Known excluded models are not included in new aggregates of historical files."""
    entry = solver_registry.get(model_id)
    return entry.notes if entry is not None and entry.status == "unavailable" else None
