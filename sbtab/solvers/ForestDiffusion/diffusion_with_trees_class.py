"""Forest-Flow / Forest-VP imported from forest_diffusion (50635ca).

See docs/GENERATIVE_ALGORITHM_AUDIT.md for corrections to the branch implementation.
Numeric inputs are used on their supplied scale; the benchmark adapter standardizes
them using training rows only. No continuous-output clipping is applied.
"""
import math
import numpy as np
from tqdm import tqdm

from sbtab.solvers.ForestDiffusion.utils.diffusion import VPSDE, get_pc_sampler
from sbtab.solvers.ForestDiffusion.utils.utils_diffusion import build_data_xt, euler_solve, IterForDMatrix, get_xt
import copy
import xgboost as xgb
from functools import partial
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from joblib import delayed, Parallel
from scipy.special import softmax

## Class for the flow-matching or diffusion model
# Categorical features should be numerical (rather than strings), make sure to use x = pd.factorize(x)[0] to make them as such
# Make sure to specific which features are categorical and which are integers
# Note: Binary features can be considered integers since they will be rounded to the nearest integer and then clipped
class ForestDiffusionModel():
  def __init__(self,
               X, # Numpy dataset
               X_covs=None, # Numpy dataset of additional covariates/features in order to sample X | X_covs (Optional); note that these variables will not be transformed, please apply your own z-scoring or min-max scaling if desired.
               label_y=None, # must be a categorical/binary variable; if provided will learn multiple models for each label y
               n_t=50, # number of noise level
               model='xgboost', # xgboost, random_forest, lgbm, catboost
               diffusion_type='flow', # vp, flow (flow is better, but only vp can be used for imputation)
               max_depth = 7, n_estimators = 100, eta=0.3, # xgboost hyperparameters
               tree_method='hist', reg_alpha=0.0, reg_lambda = 0.0, subsample=1.0, # xgboost hyperparameters
               num_leaves=31, # lgbm hyperparameters
               duplicate_K=100, # number of different noise sample per real data sample
               bin_indexes=[], # vector which indicates which column is binary
               cat_indexes=[], # vector which indicates which column is categorical (>=3 categories)
               int_indexes=[], # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
               remove_miss=False, # If True, we remove the missing values, this allow us to train the XGBoost using one model for all predictors; otherwise we cannot do it
               p_in_one=True, # When possible (when there are no missing values), will train the XGBoost using one model for all predictors
               true_min_max_values=None, # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
               gpu_hist=False, # using GPU or not with xgboost
               n_z=10, # number of noise to use in zero-shot classification
               eps=1e-3,
               beta_min=0.1,
               beta_max=8,
               n_jobs=-1, # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
               n_batch=1, # If >0 use the data iterator with the specified number of batches
               seed=666,
               **xgboost_kwargs): # you can pass extra parameter for xgboost

    assert isinstance(X, np.ndarray), "Input dataset must be a Numpy array"
    assert len(X.shape)==2, "Input dataset must have two dimensions [n,p]"
    assert diffusion_type == 'vp' or diffusion_type == 'flow'
    if X_covs is not None:
      assert X_covs.shape[0] == X.shape[0]
    if model not in ('xgboost', 'random_forest', 'lgbm', 'catboost'):
      raise ValueError('unsupported tree backend')
    if n_t < 2 or duplicate_K < 1 or n_batch < 1 or n_estimators < 1:
      raise ValueError('n_t >= 2 and positive duplicate_K, n_batch, n_estimators are required')
    if not 0 < eps < 1 or not 0 < beta_min <= beta_max:
      raise ValueError('require 0 < eps < 1 and 0 < beta_min <= beta_max')
    if np.isinf(X).any():
      raise ValueError('X must not contain infinities')
    if X_covs is not None:
      X_covs = np.asarray(X_covs)
      if X_covs.ndim != 2 or not np.isfinite(X_covs).all():
        raise ValueError('covariates must be a finite matrix')
    if label_y is not None:
      label_y = np.asarray(label_y)
      if label_y.ndim != 1 or len(label_y) != len(X):
        raise ValueError('label_y must contain one label per input row')
    self._rng = np.random.default_rng(seed)

    # Sanity check, must remove observations with only missing data
    obs_to_remove = np.isnan(X).all(axis=1)
    X = X[~obs_to_remove]
    if X_covs is not None:
      X_covs = X_covs[~obs_to_remove]
    if label_y is not None:
      label_y = label_y[~obs_to_remove]

    # Remove all missing values
    obs_to_remove = np.isnan(X).any(axis=1)
    if remove_miss or (obs_to_remove.sum() == 0):
      X = X[~obs_to_remove]
      if X_covs is not None:
        X_covs = X_covs[~obs_to_remove]
      if label_y is not None:
        label_y = label_y[~obs_to_remove]
      self.p_in_one = p_in_one # All variables p can be predicted simultaneously
    else:
      self.p_in_one = False

    if len(X) == 0 or X.shape[1] == 0 or np.isnan(X).all(axis=0).any():
      raise ValueError('every modelled column needs at least one observed training value')

    int_indexes = list(int_indexes) + list(bin_indexes) # since we round those, we do not need to dummy-code the binary variables

    if true_min_max_values is not None:
        self.X_min = true_min_max_values[0]
        self.X_max = true_min_max_values[1]
    else:
        self.X_min = np.nanmin(X, axis=0, keepdims=1)
        self.X_max = np.nanmax(X, axis=0, keepdims=1)

    self.cat_indexes = list(cat_indexes)
    self._cat_vocab = {}
    self.int_indexes = int_indexes
    self._binary_support = {i: np.unique(X[np.isfinite(X[:, i]), i]) for i in bin_indexes}
    if len(self.cat_indexes) > 0:
        X, self.X_names_before, self.X_names_after = self.dummify(X) # dummy-coding for categorical variables

    self.scaler = None

    X1 = X
    self.X_covs = X_covs
    self._n_covariates = 0 if X_covs is None else X_covs.shape[1]
    self.X1 = copy.deepcopy(X1)
    self.b, self.c = X1.shape
    if X_covs is not None:
      self.c_all = X1.shape[1] + X_covs.shape[1]
    else:
      self.c_all = X1.shape[1]
    self.n_t = n_t
    self.duplicate_K = duplicate_K
    self.model = model
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.seed = seed
    self.num_leaves = num_leaves
    self.eta = eta
    self.gpu_hist = gpu_hist
    self.label_y = label_y
    self.n_jobs = n_jobs
    self.tree_method = tree_method
    self.reg_lambda = reg_lambda
    self.reg_alpha = reg_alpha
    self.subsample = subsample
    self.n_z = n_z
    self.xgboost_kwargs = xgboost_kwargs

    if model == 'random_forest' and np.sum(np.isnan(X1)) > 0:
      raise ValueError('The dataset must not contain missing data in order to use model=random_forest')

    self.diffusion_type = diffusion_type
    self.sde = None
    self.eps = eps
    self.beta_min = beta_min
    self.beta_max = beta_max
    if diffusion_type == 'vp':
      self.sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

    self.n_batch = n_batch
    if self.label_y is not None:
      if pd.isna(self.label_y).any():
        raise ValueError('labels cannot be missing')
      self.y_uniques, counts = np.unique(self.label_y, return_counts=True)
      self.y_probs = counts / counts.sum()
    else:
      self.y_uniques, self.y_probs = np.array([0]), np.array([1.0])
    self.mask_y = {label: (self.label_y == label if self.label_y is not None else
                           np.ones(self.b, dtype=bool)) for label in self.y_uniques}
    self.t_levels = np.linspace(0.0 if diffusion_type == 'flow' else eps, 1.0, n_t)
    # n_batch is a NUMBER OF BATCHES, not a batch size. array_split covers each
    # row exactly once, including imbalanced classes and n_batch > class size.
    batches, covariates = {}, {}
    for label in self.y_uniques:
      subset = X1[self.mask_y[label]]
      count = min(int(n_batch), len(subset))
      batches[label] = np.array_split(subset, count)
      covariates[label] = (None if X_covs is None else
                           np.array_split(X_covs[self.mask_y[label]], count))

    dimensions = [None] if self.p_in_one else list(range(self.c))
    jobs = [(j, k, dim) for j in range(len(self.y_uniques))
            for k in range(n_t) for dim in dimensions]

    def fit_one(j, k, dim):
      label = self.y_uniques[j]
      return self.train_iterator(batches[label], covariates[label], self.t_levels[k], dim)

    if self.n_jobs == 1:
      fitted = [fit_one(*job) for job in tqdm(jobs, desc='Training ForestDiffusion', unit='field')]
    else:
      fitted = Parallel(n_jobs=self.n_jobs)(delayed(fit_one)(*job) for job in jobs)
    self.regr = [[None if self.p_in_one else [None] * self.c for _ in range(n_t)]
                 for _ in self.y_uniques]
    for (j, k, dim), regressor in zip(jobs, fitted):
      if dim is None:
        self.regr[j][k] = regressor
      else:
        self.regr[j][k][dim] = regressor

  def train_iterator(self, X1_splitted, X_covs_splitted, t, dim):
    it = IterForDMatrix(X1_splitted, X_covs_splitted, t=t, dim=dim,
        n_epochs=self.duplicate_K, diffusion_type=self.diffusion_type,
        eps=self.eps, sde=self.sde, seed=self.seed)
    if self.model != 'xgboost':
      # Optional sklearn backends materialize ONE time/class/coordinate, never
      # the old n_t * duplicate_K * n_rows tensor. XGBoost streams input batches.
      xs, ys = [], []
      def collect(*, data, label):
        xs.append(data)
        ys.append(label)
      while it.next(collect):
        pass
      return self.train_parallel(np.concatenate(xs), np.concatenate(ys))
    params = dict(objective='reg:squarederror', eta=self.eta, max_depth=self.max_depth,
        reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, subsample=self.subsample,
        seed=self.seed, tree_method=self.tree_method, device='cuda' if self.gpu_hist else 'cpu',
        nthread=1)
    params.update(self.xgboost_kwargs)
    if params['tree_method'] != 'hist':
      raise ValueError('QuantileDMatrix training requires tree_method=hist')
    matrix = xgb.QuantileDMatrix(it, max_bin=int(params.get('max_bin', 256)),
                                nthread=int(params['nthread']))
    if matrix.num_row() == 0:
      raise ValueError('no observed targets in this training subset')
    booster = xgb.train(params, matrix, num_boost_round=self.n_estimators)
    if self.gpu_hist and '"device":"cpu"' in booster.save_config().replace(' ', ''):
      raise RuntimeError('XGBoost silently fell back to CPU; request a CPU run or allocate a CUDA GPU')
    return booster

  def train_parallel(self, X_train, y_train):

    if self.model == 'random_forest':
      out = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.seed)
    elif self.model == 'lgbm':
      from lightgbm import LGBMRegressor
      out = LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=0.1, random_state=self.seed, force_col_wise=True)
    elif self.model == 'catboost':
      from catboost import CatBoostRegressor
      out = CatBoostRegressor(iterations=self.n_estimators, loss_function='MultiRMSE' if y_train.ndim == 2 and y_train.shape[1] > 1 else 'RMSE', max_depth=self.max_depth, silent=True,
        l2_leaf_reg=0.0, random_seed=self.seed) # consider t as a golden feature if t is a variable
    elif self.model == 'xgboost':
      out = xgb.XGBRegressor(n_estimators=self.n_estimators, objective='reg:squarederror', eta=self.eta, max_depth=self.max_depth,
        reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, subsample=self.subsample, seed=self.seed, tree_method=self.tree_method,
        device='cuda' if self.gpu_hist else 'cpu', **self.xgboost_kwargs)
    else:
      raise Exception("model value does not exists")

    if self.model == 'lgbm' and y_train.ndim == 2:
      from sklearn.multioutput import MultiOutputRegressor
      out = MultiOutputRegressor(out)
    if len(y_train.shape) == 1:
      y_no_miss = ~np.isnan(y_train)
      out.fit(X_train[y_no_miss, :], y_train[y_no_miss])
    else:
      out.fit(X_train, y_train)

    return out

  def dummify(self, X):
    """Fit category vocabularies once; preserve nonconsecutive numeric labels."""
    X = np.asarray(X, dtype=float)
    if not self._cat_vocab:
      self._input_dim = X.shape[1]
      self._numeric_indices = [i for i in range(self._input_dim) if i not in self.cat_indexes]
      self._cat_vocab = {i: np.unique(X[np.isfinite(X[:, i]), i]) for i in self.cat_indexes}
    if X.shape[1] != self._input_dim:
      raise ValueError('categorical input shape differs from training')
    parts = [X[:, self._numeric_indices]]
    names = [str(i) for i in self._numeric_indices]
    for i in self.cat_indexes:
      vocab = self._cat_vocab[i]
      if not np.isin(X[np.isfinite(X[:, i]), i], vocab).all():
        raise ValueError('unseen categorical value')
      block = (X[:, i, None] == vocab[None, :]).astype(float)
      block[np.isnan(X[:, i])] = np.nan
      parts.append(block)
      names.extend(f'{i}_{j}' for j in range(len(vocab)))
    return np.concatenate(parts, axis=1), pd.Index(map(str, range(self._input_dim))), pd.Index(names)

  def unscale(self, X):
    return self.scaler.inverse_transform(X) if self.scaler is not None else X

  def clean_onehot_data(self, X):
    if not self.cat_indexes:
      return X
    out = np.empty((len(X), self._input_dim))
    out[:, self._numeric_indices] = X[:, :len(self._numeric_indices)]
    cursor = len(self._numeric_indices)
    for i in self.cat_indexes:
      vocab = self._cat_vocab[i]
      out[:, i] = vocab[np.argmax(X[:, cursor:cursor + len(vocab)], axis=1)]
      cursor += len(vocab)
    return out

  def clip_extremes(self, X):
    if self.int_indexes is not None:
      for i in self.int_indexes:
        X[:, i] = np.round(X[:, i], decimals=0)

    for i, support in self._binary_support.items():
      X[:, i] = support[np.argmin(np.abs(X[:, i, None] - support), axis=1)]

    return X

  def _validate_covariates(self, X_covs, n):
    if self._n_covariates == 0:
      if X_covs is not None:
        raise ValueError('model was trained without covariates')
      return
    if X_covs is None or np.shape(X_covs) != (n, self._n_covariates) or not np.isfinite(X_covs).all():
      raise ValueError('provide finite covariates with the training width and one row per sample')

  def predict_over_c(self, X, i, j, k, dmat, expand=False, X_covs=None):
    if X_covs is not None:
      X = np.concatenate((X, X_covs), axis=1)

    use_dmat = isinstance(self.regr[j][i] if k is None else self.regr[j][i][k], xgb.Booster)
    if use_dmat:
      X_used = xgb.DMatrix(data=X)
    else:
      X_used = X

    if k is None:
      return np.asarray(self.regr[j][i].predict(X_used)).reshape(len(X), self.c)
    elif expand:
      return np.expand_dims(self.regr[j][i][k].predict(X_used), axis=1)  # [b, 1]
    else:
      return self.regr[j][i][k].predict(X_used)

  # Return the score-fn or ode-flow output
  def my_model(self, t, y, mask_y=None, dmat=False, unflatten=True, X_covs=None):
    if unflatten:
      # y is [b*c]
      c = self.c
      b = y.shape[0] // c
      X = y.reshape(b, c) # [b, c]
    else:
      X = y

    # Output
    out = np.zeros(X.shape) # [b, c]
    i = int(np.argmin(np.abs(self.t_levels - t)))
    for j, label in enumerate(self.y_uniques):
      if X_covs is not None:
        X_covs_masked = X_covs[mask_y[label], :]
      else:
        X_covs_masked = None
      if mask_y[label].sum() > 0:
        if self.p_in_one:
          out[mask_y[label], :] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=None, dmat=dmat, X_covs=X_covs_masked)
        else:
          for k in range(self.c):
            out[mask_y[label], k] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=k, dmat=dmat, X_covs=X_covs_masked)

    if self.diffusion_type == 'vp':
      alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
      out = - out / sigma_
    if unflatten:
      out = out.reshape(-1) # [b*c]
    return out

  # For imputation, we only give out and receive the missing values while ensuring consistency for the non-missing values
  # y0 is prior data, X_miss is real data
  def my_model_imputation(self, t, y, X_miss, sde=None, mask_y=None, dmat=False, X_covs=None, rng=None):

    if X_covs is not None:
      assert X_covs.shape[0] == X_miss.shape[0]

    y0 = (self._rng if rng is None else rng).normal(size=X_miss.shape) # Noise data
    b, c = y0.shape

    if self.diffusion_type == 'vp':
      assert sde is not None
      mean, std = sde.marginal_prob(X_miss, t)
      X = mean + std*y0 # following the sde
    else:
      X = t*X_miss + (1-t)*y0 # interpolation based on ground-truth for non-missing data
    mask_miss = np.isnan(X_miss)
    X[mask_miss] = y # replace missing data by y(t)

    # Output
    out = np.zeros(X.shape) # [b, c]
    i = int(np.argmin(np.abs(self.t_levels - t)))
    for j, label in enumerate(self.y_uniques):
      if X_covs is not None:
        X_covs_masked = X_covs[mask_y[label], :]
      else:
        X_covs_masked = None
      if mask_y[label].sum() > 0:
        if self.p_in_one:
          out[mask_y[label], :] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=None, dmat=dmat, X_covs=X_covs_masked)
        else:
          for k in range(self.c):
            out[mask_y[label], k] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=k, dmat=dmat, X_covs=X_covs_masked)

    if self.diffusion_type == 'vp':
      alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
      out = - out / sigma_

    out = out[mask_miss] # only return the missing data output
    out = out.reshape(-1) # [-1]
    return out

  # Generate new data by solving the reverse ODE/SDE
  def generate(self, batch_size=None, n_t=None, X_covs=None, max_batch_size=2048, seed=None):
    total_size = self.b if batch_size is None else int(batch_size)
    if total_size < 1 or max_batch_size < 1 or (n_t is not None and n_t < 2):
      raise ValueError('positive sample sizes and n_t >= 2 are required')
    rng = self._rng if seed is None else np.random.default_rng(seed)

    self._validate_covariates(X_covs, total_size)
    if X_covs is not None and self.label_y is not None:
      raise ValueError('generation with both labels and covariates needs p(label|covariates), which is not fitted; use joint generation or covariates alone')

    all_solutions = []
    for start_idx in range(0, total_size, max_batch_size):
      end_idx = min(start_idx + max_batch_size, total_size)
      curr_size = end_idx - start_idx

      y0 = rng.normal(size=(curr_size, self.c))

      # Generate random labels
      label_y = self.y_uniques[np.argmax(rng.multinomial(1, self.y_probs, size=y0.shape[0]), axis=1)]
      mask_y = {}
      for i in range(len(self.y_uniques)):
        mask_y[self.y_uniques[i]] = np.zeros(y0.shape[0], dtype=bool)
        mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True

      curr_X_covs = None
      if X_covs is not None:
        curr_X_covs = X_covs[start_idx:end_idx, :]

      my_model = partial(self.my_model, mask_y=mask_y, dmat=self.n_batch > 0, X_covs=curr_X_covs)

      if self.diffusion_type == 'vp':
        sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.n_t if n_t is None else n_t)
        ode_solved = get_pc_sampler(my_model, sde=sde, denoise=True, eps=self.eps, rng=rng)(y0.reshape(-1))
      else:
        ode_solved = euler_solve(my_model=my_model, y0=y0.reshape(-1), N=self.n_t if n_t is None else n_t)  # [t, b*c]

      solution = ode_solved.reshape(y0.shape[0], self.c)  # [b, c]
      solution = self.unscale(solution)
      solution = self.clean_onehot_data(solution)
      solution = self.clip_extremes(solution)

      if self.label_y is not None:
        solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1)

      all_solutions.append(solution)

    return np.concatenate(all_solutions, axis=0)

  # Impute missing data by solving the reverse ODE while keeping the non-missing data intact
  def impute(self, k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None, X_covs=None, seed=None): # X is data with missing values
    if self.diffusion_type != 'vp' or self.cat_indexes:
      raise ValueError('imputation supports only VP with numeric inputs; encode categorical columns externally')
    if k < 1:
      raise ValueError('k must be positive')
    rng = self._rng if seed is None else np.random.default_rng(seed)

    using_training_rows = X is None
    if X is None:
      X = self.X1
      if X_covs is None:
        X_covs = self.X_covs
    if label_y is None and using_training_rows:
      label_y = self.label_y
    if X is None or np.ndim(X) != 2 or X.shape[1] != self.c or np.isinf(X).any():
      raise ValueError('X must have the training width and contain finite values or NaN')
    if self.label_y is not None:
      if label_y is None or np.shape(label_y) != (len(X),) or not np.isin(label_y, self.y_uniques).all():
        raise ValueError('provide one known training label per imputation row')
    elif label_y is not None:
      raise ValueError('model was trained without label conditioning')
    if n_t is None:
      n_t = self.n_t

    self._validate_covariates(X_covs, len(X))

    if self.diffusion_type == 'vp':
      sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

    if label_y is None: # single category 0
      mask_y = {}
      mask_y[0] = np.ones(X.shape[0], dtype=bool)
    else:
      mask_y = {} # mask for which observations has a specific value of y
      for i in range(len(self.y_uniques)):
        mask_y[self.y_uniques[i]] = np.zeros(X.shape[0], dtype=bool)
        mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True

    my_model_imputation = partial(self.my_model_imputation, X_miss=X, sde=sde, mask_y=mask_y, dmat=self.n_batch > 0, X_covs=X_covs, rng=rng)

    for i in range(k):
      y0 = rng.normal(size=X.shape)

      mask_miss = np.isnan(X)
      y0_miss = y0[mask_miss].reshape(-1)
      solution = copy.deepcopy(X) # Solution start with dataset which contains some missing values
      if mask_miss.any():
        ode_solved = get_pc_sampler(my_model_imputation, sde=sde, denoise=True, repaint=repaint, eps=self.eps, rng=rng)(y0_miss, r=r, j=int(math.ceil(j*n_t)))
        solution[mask_miss] = ode_solved # replace missing values with imputed values
      solution = self.unscale(solution)
      solution = self.clean_onehot_data(solution)
      solution = self.clip_extremes(solution)
      # Concatenate y label if needed
      if self.label_y is not None:
        solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1)
      if i == 0:
        imputed_data = np.expand_dims(solution, axis=0)
      else:
        imputed_data = np.concatenate((imputed_data, np.expand_dims(solution, axis=0)), axis=0)
    return imputed_data[0] if k==1 else imputed_data

  # Zero-shot classification of one batch
  def zero_shot_classification(self, x, n_t=10, n_z=10, X_covs=None):
    assert self.label_y is not None # must have label conditioning to work
    self._validate_covariates(X_covs, len(x))
    if n_t < 2 or n_z < 1 or not np.isfinite(x).all():
      raise ValueError('classification requires finite inputs, n_t >= 2 and n_z >= 1')

    h = 1 / n_t
    num_classes = len(self.y_uniques)
    L2_dist = []
    for i in range(num_classes): # for each class

      # Class conditioning
      mask_y = {}
      for k in range(len(self.y_uniques)):
        if k == i:
          mask_y[self.y_uniques[k]] = np.ones(x.shape[0], dtype=bool)
        else:
          mask_y[self.y_uniques[k]] = np.zeros(x.shape[0], dtype=bool)

      L2_dist_ = []
      for k in range(n_z): # monte-carlo over multiple noises
        t = 0
        for j in range(n_t-1): # averaging over multiple noise levels [t=1/n, ... (n-1)/n]
          t = t + h
          y0 = np.random.default_rng(10000*k + j).normal(size=x.shape)
          xt = get_xt(x1=x, t=t, x0=y0, dim=None, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)[0]
          pred_ = self.my_model(t=t, y=xt, mask_y=mask_y, unflatten=False, dmat=self.n_batch > 0, X_covs=X_covs)
          if self.diffusion_type == 'flow':
            x0 = x - pred_ # x0 = x1 - (x1 - x0)
          elif self.diffusion_type == 'vp':
            _, std = self.sde.marginal_prob_coef(xt, t)
            x0 = -std * pred_  # my_model returns a score, classifier compares NOISE
          L2_dist_ += [np.expand_dims(np.sum((x0 - y0) ** 2, axis=1), axis=0)] # [1, b]
      L2_dist += [np.concatenate(L2_dist_, axis=0)] # [n_z*n_t, b]

    # Based on absolute
    L2_abs = []
    for i in range(num_classes): # for each class
      L2_abs += [np.expand_dims(np.mean(L2_dist[i], axis=0), axis=0)] # [1, b]
    L2_abs = np.concatenate(L2_abs, axis=0) # [c, b]
    prob_avg = softmax(-L2_abs, axis=0) # [b]
    most_likely_class_avg = np.argmin(L2_abs, axis=0) # [b]
    return self.y_uniques[most_likely_class_avg], prob_avg.T

  # Zero-shot classification using https://diffusion-classifier.github.io/static/docs/DiffusionClassifier.pdf
  # Return the absolute and relative accuracies
  def predict(self, X, n_t=None, n_z=None, X_covs=None):
    if n_t is None:
      n_t = self.n_t
    if n_z is None:
      n_z = self.n_z

    # Data transformation (assuming we get the raw data)
    if len(self.cat_indexes) > 0:
      X, _, _ = self.dummify(X) # dummy-coding for categorical variables
    if self.scaler is not None:
      X = self.scaler.transform(X)

    most_likely_class_avg, prob_avg = self.zero_shot_classification(X, n_t=n_t, n_z=n_z, X_covs=X_covs)

    return most_likely_class_avg

  def predict_proba(self, X, n_t=None, n_z=None, X_covs=None):
    if n_t is None:
      n_t = self.n_t
    if n_z is None:
      n_z = self.n_z

    # Data transformation (assuming we get the raw data)
    if len(self.cat_indexes) > 0:
      X, _, _ = self.dummify(X) # dummy-coding for categorical variables
    if self.scaler is not None:
      X = self.scaler.transform(X)

    most_likely_class_avg, prob_avg = self.zero_shot_classification(X, n_t=n_t, n_z=n_z, X_covs=X_covs)

    return prob_avg
