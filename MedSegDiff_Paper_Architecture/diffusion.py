import enum
import numpy as np
import torch

class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()
    VELOCITY = enum.auto()

class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(name, steps):
    if name == "linear":
        return np.linspace(1e-4, 0.02, steps)
    if name == "cosine":
        return betas_for_alpha_bar(
            steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError()

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)

    all_steps = []
    start = 0

    for i, count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        stride = max(size // count, 1)
        steps = list(range(start, start + size, stride))[:count]
        all_steps.extend(steps)
        start += size

    return sorted(all_steps)

def _extract(a, t, shape):
    out = a.gather(0, t)
    return out.view(t.shape[0], *((1,) * (len(shape) - 1)))

class GaussianDiffusion:
    def __init__(
        self,
        betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = torch.tensor(betas, dtype=torch.float64)
        self.betas = betas
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        var = _extract(self.posterior_variance, t, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, x0):
        return (
            x_t - _extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x0
        ) / _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    
    def p_mean_variance(self, model, x, t, cond=None):
        if hasattr(self, "timestep_map"):
            t_for_model = self.timestep_map.to(t.device)[t]
        else:
            t_for_model = t

        model_out = model(x, t, cond)

        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            model_out, var_values = torch.split(model_out, x.shape[1], dim=1)
            min_log = _extract(self.posterior_log_variance_clipped, t, x.shape)
            max_log = torch.log(self.betas)[t].view(-1, 1, 1, 1)
            frac = (var_values + 1) / 2
            model_log_var = frac * max_log + (1 - frac) * min_log
        else:
            model_log_var = _extract(self.posterior_log_variance_clipped, t, x.shape)

        if self.model_mean_type == ModelMeanType.EPSILON:
            x_start = self._predict_xstart_from_eps(x, t, model_out)
        elif self.model_mean_type == ModelMeanType.START_X:
            x_start = model_out
        else:
            raise NotImplementedError()

        mean, _, _ = self.q_posterior_mean_variance(x_start, x, t)
        return mean, model_log_var, x_start
    
    def p_sample(self, model, x, t, cond=None):
        mean, log_var, x_start = self.p_mean_variance(model, x, t, cond)
        noise = torch.randn_like(x)
        nonzero = (t != 0).float().view(-1, 1, 1, 1)
        sample = mean + nonzero * torch.exp(0.5 * log_var) * noise
        return sample, x_start
    
    
class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, betas, **kwargs):
        self.use_timesteps = sorted(use_timesteps)
        self.timestep_map = []

        base_diffusion = GaussianDiffusion(betas=betas, **kwargs)
        last_alpha = 1.0
        new_betas = []

        for i, alpha in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha / last_alpha)
                last_alpha = alpha
                self.timestep_map.append(i)

        super().__init__(betas=new_betas, **kwargs)

    def p_sample_loop(self, model, shape, cond=None, device="cuda"):
        x = torch.randn(shape, device=device)

        for i in reversed(range(len(self.use_timesteps))):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(model, x, t, cond)

        return x