"""Default matrix: one basic joint MLP per DSB family; explicit variants remain available."""
from sbtab.solvers.registry import get_entry


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
    return get_entry(model_id)
