import numpy as np
import abc
import functools

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, rng, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(y=x, t=t)
        drift = drift - (diffusion ** 2)*(score * (0.5 if self.probability_flow else 1.))
        # Set the diffusion function to zero for ODEs.
        diffusion = np.zeros_like(diffusion) if self.probability_flow else diffusion
        return drift, diffusion

    return RSDE()

class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    grid = np.linspace(0.0, 1.0, N + 1)
    integral = beta_min * grid + 0.5 * (beta_max - beta_min) * grid ** 2
    self.discrete_betas = -np.expm1(-np.diff(integral))
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t * x
    diffusion = np.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = np.exp(log_mean_coeff)*x
    std = np.sqrt(1 - np.exp(2. * log_mean_coeff))
    return mean, std

  def marginal_prob_coef(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = np.exp(log_mean_coeff)
    std = np.sqrt(1 - np.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return np.random.normal(size=shape)

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    pass

class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, rng=None):
    super().__init__(sde, score_fn)
    self.rng = np.random if rng is None else rng

  def update_fn(self, x, t, h):
    my_sde = self.rsde.sde
    z = self.rng.normal(size=x.shape)
    drift, diffusion = my_sde(x, t)
    x_mean = x - drift * h
    x = x_mean + diffusion*np.sqrt(h)*z
    return x, x_mean

def shared_predictor_update_fn(x, t, h=None, sde=None, score_fn=None, rng=None):
  """A wrapper that configures and returns the update function of predictors."""
  predictor_obj = EulerMaruyamaPredictor(sde, score_fn, rng=rng)
  return predictor_obj.update_fn(x, t, h)

# VP sampler

def get_pc_sampler(score_fn, sde, denoise=True, eps=1e-3, repaint=False, rng=None):

  if sde.N < 2 or not 0 < eps < sde.T:
    raise ValueError('VP sampling needs N >= 2 and 0 < eps < T')
  rng = np.random if rng is None else rng

  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          score_fn=score_fn, rng=rng)

  def pc_sampler(prior, r=5, j=5):
    # Initial sample
    x = prior
    timesteps = np.linspace(sde.T, eps, sde.N)
    h = timesteps - np.append(timesteps, 0)[1:] # true step-size: difference between current time and next time (only the new predictor classes will use h, others will ignore)
    N = sde.N - 1

    for i in range(N):
      x, x_mean = predictor_update_fn(x, timesteps[i], h[i])

    if denoise: # Tweedie formula
      alpha, std = sde.marginal_prob_coef(x, eps)
      x = (x + (std ** 2)*score_fn(y=x, t=eps)) / alpha

    return x

  def pc_sampler_repaint(prior, r=5, j=5):
    if r < 1 or j < 1:
      raise ValueError('repaint r and j must be positive')
    # Initial sample
    x = prior
    timesteps = np.linspace(sde.T, eps, sde.N)
    h = timesteps - np.append(timesteps, 0)[1:] # true step-size: difference between current time and next time (only the new predictor classes will use h, others will ignore)
    N = sde.N - 1

    i_repaint = 0
    i = 0
    while i < N:
      x, x_mean = predictor_update_fn(x, timesteps[i], h[i])
      if i_repaint < r-1 and (i+1) % j == 0: # we did j iterations, but not enough repaint, we must repaint again
        # Going backward in time; using Euler-Maruyama
        # Exact VP transition from the state time AFTER the reverse step to
        # the beginning of this repaint block (not a large Euler jump).
        low, high = timesteps[i + 1], timesteps[i - j + 1]
        a_low, _ = sde.marginal_prob_coef(x, low)
        a_high, _ = sde.marginal_prob_coef(x, high)
        ratio = a_high / a_low
        x = ratio * x + np.sqrt(1.0 - ratio ** 2) * rng.normal(size=x.shape)
        # iterate back
        i_repaint = i_repaint + 1
        i = i - j
      elif i_repaint == r-1 and (i+1) % j == 0: # we did j iterations and enough repaint, we continue and reset the repaint counter
        i_repaint = 0
      i = i + 1

    if denoise: # Tweedie formula
      alpha, std = sde.marginal_prob_coef(x, eps)
      x = (x + (std ** 2)*score_fn(y=x, t=eps)) / alpha

    return x

  if repaint:
    return pc_sampler_repaint
  else:
    return pc_sampler
