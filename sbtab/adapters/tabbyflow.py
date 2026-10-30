"""Canonical joint TabbyFlow adapter: explicit schema, fresh fits, complete inference checkpoints."""
from sbtab.adapters.baselines import _BaselineAdapter
from sbtab.data.schema import TabularSchema


class TabbyFlowAdapter(_BaselineAdapter):
    registry_id = "tabbyflow"
    MODEL_FILE = "tabbyflow.pt"
    DEFAULTS = dict(max_train_steps=2000, batch_size=1024, n_frequencies=512, lr=1e-3,
                    weight_decay=0.0, cond_vel="ot", ode_solver="euler", ode_steps=100,
                    scheduler_factor=0.9, scheduler_patience_epochs=20,
                    early_stopping_patience_epochs=200, sample_batch_size=512, device="cpu")

    @classmethod
    def resolve_config(cls, config):
        from sbtab.baselines.tabbyflow import TabbyFlowConfig
        effective = super().resolve_config(config)
        # The search-space loader probes consumed keys with None placeholders.
        # Validate fixed values now; concrete sampled values are validated again.
        checked = {k: cls.DEFAULTS[k] if v is None else v for k, v in effective.items()}
        TabbyFlowConfig(**checked, seed=0)
        return effective

    def _wrapper_class(self):
        from sbtab.baselines.tabbyflow import TabbyFlowSynthesizer
        return TabbyFlowSynthesizer

    def _make_wrapper(self):
        from sbtab.baselines.tabbyflow import TabbyFlowConfig
        return self._wrapper_class()(TabbyFlowConfig(**self.config, seed=int(self.seed)))

    def _fit_model(self):
        s = self.schema
        # Explicit groups include the target in its declared type. The wrapper's
        # role resolver accepts that without dtype/cardinality inference.
        schema = TabularSchema(continuous_cols=list(s.continuous), discrete_cols=list(s.discrete),
                               categorical_cols=list(s.categorical), target_col=s.target)
        self.model.fit(self._encoded, schema=schema, task_type=s.task or "regression")

    def _describe_model(self):
        from sbtab.baselines.tabbyflow.model import TABBYFLOW_OFFICIAL_REPOSITORY, TABBYFLOW_OFFICIAL_VARIANT
        return {"source": TABBYFLOW_OFFICIAL_REPOSITORY, "source_variant": TABBYFLOW_OFFICIAL_VARIANT,
                "representation": "train-fitted uniform quantiles for numerical columns; full nominal one-hot",
                "discrete_decoding": "nearest training support", "numeric_inverse": "empirical quantile inverse (training range)",
                "conditional_path": self.config["cond_vel"], "ode_solver": self.config["ode_solver"],
                "ode_steps": self.config["ode_steps"], "n_updates": self.n_updates,
                "best_training_loss": self.model.best_train_loss_, "checkpoint_contains_training_rows": False,
                "source_approximation": self.config["cond_vel"].lower() in {"vp", "ve"},
                "checkpoint_kind": "inference; optimizer and scheduler are not resumed"}
