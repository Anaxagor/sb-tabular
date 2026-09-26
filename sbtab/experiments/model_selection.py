"""Model scope for the experiment: one basic joint MLP per DSB/DSBM family."""
from sbtab.experiments.experiment_common import StageError
from sbtab.solvers.registry import get_entry


BASIC_DSB_MODELS = {
    "IPF-DSB": "dsb_ct_joint_mlp",
    "IMF-DSBM": "dsbm_ct_joint_mlp",
}


def exclusion_reason(model_id: str):
    entry = get_entry(model_id)
    selected = BASIC_DSB_MODELS.get(entry.family)
    if selected is not None and model_id != selected:
        return f"excluded from experiments: {entry.family} uses only the basic joint MLP {selected!r}"
    return None


def require_experiment_model(model_id: str):
    reason = exclusion_reason(model_id)
    if reason:
        raise StageError("not_applicable", f"model {model_id!r}: {reason}")
    return get_entry(model_id)
