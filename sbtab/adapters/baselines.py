"""
Adapters for the baseline wrappers under sbtab/baselines.

The wrappers receive the training rows in the COMMON representation together with
EXPLICIT column roles from the schema, so no dtype/cardinality inference can turn
integer label codes into a continuous feature. Any further learned transform
(z-scoring of the Gaussian block, one-hot codecs, SDV's transformers) lives inside
the wrapper, is fitted on the same rows, is serialised with it, and is timed as
part of generator fitting. Output is mapped back to the common schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from sbtab.adapters.base import ModelAdapter


class _BaselineAdapter(ModelAdapter):
    native_regimes = ("continuous", "discrete", "mixed")
    MODEL_FILE: ClassVar[str] = "model.bin"

    def _wrapper_class(self):
        raise NotImplementedError

    def _make_wrapper(self):
        raise NotImplementedError

    def _roles(self) -> dict:
        s = self.schema
        return dict(continuous_cols=list(s.continuous), discrete_cols=list(s.discrete),
                    categorical_cols=list(s.categorical), target_col=s.target, task=s.task)

    def _prepare(self, train: pd.DataFrame) -> None:
        self._encoded = train.copy()
        self._dtypes = {c: str(train[c].dtype) for c in train.columns}

    def _build_model(self) -> None:
        self.model = self._make_wrapper()

    def _fit_model(self) -> None:
        self.model.fit(self._encoded, **self._roles())

    def _generate(self, n: int, seed: int) -> pd.DataFrame:
        return self.model.sample(n, seed=seed)

    def _decode(self, generated: pd.DataFrame) -> pd.DataFrame:
        df = generated[self.schema.column_order].reset_index(drop=True).copy()
        for c in self.schema.categorical:            # codes stay integral; anything else is reported as invalid downstream
            v = pd.to_numeric(df[c], errors="coerce")
            df[c] = v.astype(np.int64) if v.notna().all() and np.all(np.mod(v, 1) == 0) else v
        for c in self.schema.continuous + self.schema.discrete:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        self.decoding_report_ = dict(getattr(self.model, "decoding_report_", {}) or {})
        for extra in ("label_report_", "sample_report_", "last_sample_cost_"):
            if getattr(self.model, extra, None):
                self.decoding_report_[extra.strip("_")] = getattr(self.model, extra)
        return df

    @property
    def n_updates(self):
        return getattr(self.model, "n_updates", None)

    def describe(self) -> dict:
        m = self.model
        d = {"variant_id": getattr(m, "variant_id", self.registry_id), "role_source": getattr(m, "role_source_", None),
             "target_handling": "joint row (X, y); unconditional", "representation": "common + wrapper-internal codec",
             "label_distribution": getattr(m, "label_distribution_", None)}
        d.update(self._describe_model())
        return d

    def _describe_model(self) -> dict:
        return {}

    def _state(self) -> dict:
        return {"dtypes": self._dtypes}

    def _save_model(self, directory: Path) -> None:
        self.model.save_checkpoint(directory / self.MODEL_FILE)

    def _load_model(self, directory: Path, state: dict) -> None:
        self._dtypes = dict(state.get("dtypes", {}))
        self.model = self._wrapper_class().load_checkpoint(directory / self.MODEL_FILE)


class TabDDPMAdapter(_BaselineAdapter):
    """``steps`` is THE training budget (optimizer updates); there is no second, competing budget key."""
    registry_id = "tabddpm"
    DEFAULTS = dict(steps=10000, num_timesteps=1000, batch_size=4096, lr=1e-3, weight_decay=1e-4, n_layers=4,
                    layer_size=256, dropout=0.0, scheduler="cosine", gaussian_loss_type="mse", ema_decay=0.999,
                    ema_warmup=True, device="cpu")

    def _wrapper_class(self):
        from sbtab.baselines.tabddpm import TabDDPMWrapper
        return TabDDPMWrapper

    def _make_wrapper(self):
        from sbtab.baselines.tabddpm import TabDDPMConfig
        c = self.config
        cfg = TabDDPMConfig(steps=int(c["steps"]), n_epochs=None, num_timesteps=int(c["num_timesteps"]),
                            batch_size=int(c["batch_size"]), lr=float(c["lr"]), weight_decay=float(c["weight_decay"]),
                            d_layers=[int(c["layer_size"])] * int(c["n_layers"]), dropout=float(c["dropout"]),
                            gaussian_loss_type=str(c["gaussian_loss_type"]), scheduler=str(c["scheduler"]),
                            ema_decay=float(c["ema_decay"]), ema_warmup=bool(c["ema_warmup"]), device=str(c["device"]),
                            seed=int(self.seed))
        return self._wrapper_class()(cfg)

    def _describe_model(self) -> dict:
        return {"total_steps": getattr(self.model, "total_steps_", None), "sampling": "EMA weights, eval mode, no clipping",
                "final_loss": getattr(self.model, "final_loss_", None)}


class VEScoreSDEAdapter(_BaselineAdapter):
    """A simplified VE score-SDE model. It is NOT a faithful STaSy (see sbtab.baselines.stasy.FAITHFULNESS)."""
    registry_id = "ve_score_sde_simplified"
    DEFAULTS = dict(steps=5000, batch_size=512, lr=2e-4, weight_decay=0.0, hidden_dim=256, n_layers=4, time_emb_dim=64,
                    dropout=0.0, sigma_min=0.01, sigma_max=50.0, n_sampling_steps=1000, n_corrector_steps=1,
                    corrector_snr=0.16, time_curriculum=False, grad_clip=1.0, device="cpu")

    def _wrapper_class(self):
        from sbtab.baselines.stasy import VEScoreSDEBaseline
        return VEScoreSDEBaseline

    def _make_wrapper(self):
        from sbtab.baselines.stasy import VEScoreSDEConfig
        c = self.config
        cfg = VEScoreSDEConfig(hidden_dim=int(c["hidden_dim"]), n_layers=int(c["n_layers"]), time_emb_dim=int(c["time_emb_dim"]),
                               dropout=float(c["dropout"]), sigma_min=float(c["sigma_min"]), sigma_max=float(c["sigma_max"]),
                               n_epochs=None, steps=int(c["steps"]), batch_size=int(c["batch_size"]), lr=float(c["lr"]),
                               weight_decay=float(c["weight_decay"]), grad_clip=None if c["grad_clip"] is None else float(c["grad_clip"]),
                               time_curriculum=bool(c["time_curriculum"]), n_sampling_steps=int(c["n_sampling_steps"]),
                               n_corrector_steps=int(c["n_corrector_steps"]), corrector_snr=float(c["corrector_snr"]),
                               device=str(c["device"]), seed=int(self.seed))
        return self._wrapper_class()(cfg)

    def _describe_model(self) -> dict:
        from sbtab.baselines.stasy import FAITHFULNESS
        return {"faithfulness": FAITHFULNESS, "sampler": "predictor-corrector, VE SDE"}


class CTGANAdapter(_BaselineAdapter):
    registry_id = "ctgan"
    MODEL_FILE = "model_dir"
    DEFAULTS = dict(epochs=300, batch_size=500, pac=10, embedding_dim=128, generator_width=256, discriminator_width=256,
                    generator_lr=2e-4, discriminator_lr=2e-4, discriminator_steps=1, enable_gpu=False)

    def _wrapper_class(self):
        from sbtab.baselines.ctgan import CTGANWrapper
        return CTGANWrapper

    def _make_wrapper(self):
        from sbtab.baselines.ctgan import CTGANConfig
        c = self.config
        cfg = CTGANConfig(embedding_dim=int(c["embedding_dim"]), generator_dim=(int(c["generator_width"]),) * 2,
                          discriminator_dim=(int(c["discriminator_width"]),) * 2, generator_lr=float(c["generator_lr"]),
                          discriminator_lr=float(c["discriminator_lr"]), discriminator_steps=int(c["discriminator_steps"]),
                          batch_size=int(c["batch_size"]), epochs=int(c["epochs"]), pac=int(c["pac"]),
                          enforce_min_max_values=False,        # SDV's default CLIPS to the training range: prohibited
                          enforce_rounding=False, enable_gpu=bool(c["enable_gpu"]), seed=int(self.seed))
        return self._wrapper_class()(cfg)

    def _describe_model(self) -> dict:
        return {"metadata": getattr(self.model, "metadata_dict_", None), "clipping": "disabled (enforce_min_max_values=False)",
                "n_updates_note": "declared count epochs * max(N // batch, 1), not a measurement"}


class TabPFGenAdapter(_BaselineAdapter):
    """
    Pretrained-prior wrapper: fit() performs NO gradient updates (it stores the conditioning
    rows, which ARE the model); all cost is paid at sampling time and is reported in
    decoding_report_['last_sample_cost']. Labels are resampled to the saved TRAINING
    label distribution — never balanced, never uniform.
    """
    registry_id = "tabpfgen"
    supported_regimes = ("continuous", "discrete", "mixed")
    DEFAULTS = dict(n_sgld_steps=1000, sgld_step_size=0.01, sgld_noise_scale=0.01, use_quantiles=True,
                    prior_oversample=2.0, max_topup_rounds=5, device="cpu")

    def _wrapper_class(self):
        from sbtab.baselines.tabpfn import TabPFGenGenerative
        return TabPFGenGenerative

    def _make_wrapper(self):
        from sbtab.baselines.tabpfn import TabPFGenConfig
        if self.schema.target is None:
            raise ValueError("TabPFGen needs a declared target column")
        c = self.config
        cfg = TabPFGenConfig(target_col=self.schema.target, task=self.schema.task, n_sgld_steps=int(c["n_sgld_steps"]),
                             sgld_step_size=float(c["sgld_step_size"]), sgld_noise_scale=float(c["sgld_noise_scale"]),
                             device=str(c["device"]), balance_classes=False, use_quantiles=bool(c["use_quantiles"]),
                             seed=int(self.seed), preserve_label_prior=True, prior_oversample=float(c["prior_oversample"]),
                             max_topup_rounds=int(c["max_topup_rounds"]))
        return self._wrapper_class()(cfg)

    def _describe_model(self) -> dict:
        m = self.model
        return {"adaptation_cost": getattr(m, "adaptation_cost_", None), "pretrained_identity": getattr(m, "pretrained_identity_", None),
                "conditioning_context": getattr(m, "conditioning_context_", None), "sgld": getattr(m, "sgld_settings_", None),
                "checkpoint_contains_training_rows": True}
