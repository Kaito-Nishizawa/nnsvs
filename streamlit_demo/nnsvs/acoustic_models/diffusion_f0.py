from collections import deque
from functools import partial

import numpy as np
import torch
from nnsvs.base import BaseModel, PredictionType
from nnsvs.diffsinger.diffusion import (
    cosine_beta_schedule,
    extract,
    linear_beta_schedule,
    noise_like,
)
from tqdm import tqdm

beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


def clip_lf0_residual_(lf0_residual, residual_f0_max_cent=600):
    # Bound the output to [-0.35, 0.35] when residual_f0_max_cent = 600
    max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200
    lf0_residual.clamp_(-max_lf0_ratio, max_lf0_ratio)
    return lf0_residual


class ResF0GaussianDiffusion(BaseModel):
    def __init__(
        self,
        in_dim,
        out_dim,
        denoise_fn,
        encoder=None,
        K_step=100,
        betas=None,
        schedule_type="linear",
        scheduler_params=None,
        pndm_speedup=None,
        objective="pred_noise",
        prior_std=0.12,
        prior_std_inf=0.12,
        in_lf0_idx=51,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        in_lf0_mean=None,
        in_lf0_scale=None,
        out_lf0_idx=0,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        residual_f0_max_cent=600,
        hard_clip_residual_f0=False,
        do_classifier_free_guidance=True,
        guidance_scale=1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.denoise_fn = denoise_fn
        self.K_step = K_step
        self.pndm_speedup = pndm_speedup
        self.prior_std = prior_std
        self.prior_std_inf = prior_std_inf
        self.encoder = encoder
        self.objective = objective
        if scheduler_params is None:
            if schedule_type == "linear":
                scheduler_params = {"max_beta": 0.06}
            elif schedule_type == "cosine":
                scheduler_params = {"s": 0.008}
        self.in_lf0_idx = in_lf0_idx
        # for min-max normalized lf0
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        # for mean-var normalized lf0
        self.in_lf0_mean = in_lf0_mean
        self.in_lf0_scale = in_lf0_scale
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.residual_f0_max_cent = residual_f0_max_cent
        self.hard_clip_residual_f0 = hard_clip_residual_f0
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale

        if do_classifier_free_guidance and objective != "pred_noise":
            raise ValueError(
                "do_classifier_free_guidance is only valid for pred_noise objective"
            )

        if encoder is not None:
            assert encoder.in_dim == in_dim, "encoder input dim must match in_dim"
        assert out_dim == denoise_fn.in_dim, "denoise_fn input dim must match out_dim"

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            betas = beta_schedule[schedule_type](K_step, **scheduler_params)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0
        # at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def requires_spk(self):
        return self.denoise_fn.requires_spk()

    def _norm(self, lf0, lf0_score_denorm):
        lf0_denorm = lf0 * self.out_lf0_scale + self.out_lf0_mean
        lf0_residual = lf0_denorm - lf0_score_denorm
        if self.hard_clip_residual_f0:
            lf0_residual = clip_lf0_residual_(lf0_residual, self.residual_f0_max_cent)
        return lf0_residual

    def _denorm(self, lf0_residual, lf0_score_denorm):
        lf0_pred_denorm = lf0_residual + lf0_score_denorm
        lf0_pred = (lf0_pred_denorm - self.out_lf0_mean) / self.out_lf0_scale
        return lf0_pred

    def prediction_type(self):
        return PredictionType.DIFFUSION

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool, spk=None):
        if self.objective == "pred_noise":
            if self.do_classifier_free_guidance:
                x = torch.cat([x] * 2)
                cond = torch.cat([cond] * 2)
                # NOTE: Use zero vectors to indicate null speaker embedding
                spk = torch.cat([torch.zeros_like(spk), spk])
                t = torch.cat([t] * 2)

            pred_noise = self.denoise_fn(x, t, cond=cond, spk=spk)

            if self.do_classifier_free_guidance:
                pred_noise_uncond, pred_noise = pred_noise.chunk(2, dim=0)
                pred_noise = pred_noise_uncond + self.guidance_scale * (
                    pred_noise - pred_noise_uncond
                )
                x = x.chunk(2, dim=0)[0]
                t = t.chunk(2, dim=0)[0]

            pred_x_start = self.predict_start_from_noise(x, t=t, noise=pred_noise)
            if clip_denoised:
                pred_x_start = clip_lf0_residual_(
                    pred_x_start, self.residual_f0_max_cent
                )
        elif self.objective == "pred_x0":
            pred_x_start = self.denoise_fn(x, t, cond=cond, spk=spk)
            if clip_denoised:
                pred_x_start = clip_lf0_residual_(
                    pred_x_start, self.residual_f0_max_cent
                )
            pred_noise = self.predict_noise_from_start(x, t, x0=pred_x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=pred_x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        cond,
        noise_fn=torch.randn,
        clip_denoised=True,
        repeat_noise=False,
        spk=None,
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, cond=cond, clip_denoised=clip_denoised, spk=spk
        )
        noise = noise_like(x.shape, noise_fn, device, repeat_noise) * self.prior_std
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, spk=None):
        """
        Use the PLMS method from Pseudo Numerical Methods for Diffusion Models on Manifolds
        https://arxiv.org/abs/2202.09778.
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(
                self.alphas_cumprod,
                torch.max(t - interval, torch.zeros_like(t)),
                x.shape,
            )
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1
                / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
                * noise_t
            )
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        if self.objective == "pred_noise":
            noise_pred = self.denoise_fn(x, t, cond=cond, spk=spk)
        elif self.objective == "pred_x0":
            pred_x_start = self.denoise_fn(x, t, cond=cond, spk=spk)
            noise_pred = self.predict_noise_from_start(x, t, x0=pred_x_start)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(
                x_pred, torch.max(t - interval, torch.zeros_like(t)), cond=cond, spk=spk
            )
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (
                23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]
            ) / 12
        elif len(noise_list) >= 3:
            noise_pred_prime = (
                55 * noise_pred
                - 59 * noise_list[-1]
                + 37 * noise_list[-2]
                - 9 * noise_list[-3]
            ) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start) * self.prior_std
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, cond, lengths=None, y=None, spk=None, spk_enc_inp=None):
        """Forward step

        Args:
            cond (torch.Tensor): conditioning features of shaep (B, T, encoder_hidden_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch
            y (torch.Tensor): ground truth of shape (B, T, C)

        Returns:
            tuple of tensors (B, T, in_dim), (B, T, in_dim)
        """
        B = cond.shape[0]
        device = cond.device

        # Denormalize lf0 from input musical score
        lf0_score = cond[:, :, self.in_lf0_idx].unsqueeze(-1)

        in_lf0_meanvar_norm = (
            self.in_lf0_mean is not None and self.in_lf0_scale is not None
        )

        if in_lf0_meanvar_norm:
            lf0_score_denorm = lf0_score * self.in_lf0_scale + self.in_lf0_mean
        else:
            lf0_score_denorm = (
                lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min
            )

        if self.encoder is not None:
            if self.encoder.requires_spk():
                cond = self.encoder(cond, lengths, spk=spk, spk_enc_inp=spk_enc_inp)
            else:
                cond = self.encoder(cond, lengths)

        # (B, M, T)
        cond = cond.transpose(1, 2)

        t = torch.randint(0, self.K_step, (B,), device=device).long()
        x_start = self._norm(y, lf0_score_denorm)
        x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]

        noise = torch.randn_like(x_start) * self.prior_std
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoise_fn(x_noisy, t, cond, spk=spk)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start

        target = target.squeeze(1).transpose(1, 2)
        model_out = model_out.squeeze(1).transpose(1, 2)

        return target, model_out

    def inference(self, cond, lengths=None, spk=None, spk_enc_inp=None):
        B = cond.shape[0]
        device = cond.device

        # Denormalize lf0 from input musical score
        lf0_score = cond[:, :, self.in_lf0_idx].unsqueeze(-1)

        in_lf0_meanvar_norm = (
            self.in_lf0_mean is not None and self.in_lf0_scale is not None
        )

        if in_lf0_meanvar_norm:
            lf0_score_denorm = lf0_score * self.in_lf0_scale + self.in_lf0_mean
        else:
            lf0_score_denorm = (
                lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min
            )

        if self.encoder is not None:
            if self.encoder.requires_spk():
                cond = self.encoder(cond, lengths, spk=spk, spk_enc_inp=spk_enc_inp)
            else:
                cond = self.encoder(cond, lengths)

        # (B, M, T)
        cond = cond.transpose(1, 2)

        # Replace prior std with prior std for inference
        _prior_std_org = self.prior_std
        self.prior_std = self.prior_std_inf

        t = self.K_step
        shape = (cond.shape[0], 1, self.out_dim, cond.shape[2])
        x = torch.randn(shape, device=device) * self.prior_std

        if self.pndm_speedup:
            self.noise_list = deque(maxlen=4)
            iteration_interval = int(self.pndm_speedup)
            for i in tqdm(
                reversed(range(0, t, iteration_interval)),
                desc="sample time step",
                total=t // iteration_interval,
            ):
                x = self.p_sample_plms(
                    x,
                    torch.full((B,), i, device=device, dtype=torch.long),
                    iteration_interval,
                    cond,
                    spk=spk,
                )
                x = clip_lf0_residual_(x, self.residual_f0_max_cent)

        else:
            for i in tqdm(reversed(range(0, t)), desc="sample time step", total=t):
                x = self.p_sample(
                    x,
                    torch.full((B,), i, device=device, dtype=torch.long),
                    cond,
                    spk=spk,
                )
                x = clip_lf0_residual_(x, self.residual_f0_max_cent)

        self.prior_std = _prior_std_org

        x = self._denorm(x[:, 0].transpose(1, 2), lf0_score_denorm)
        return x
