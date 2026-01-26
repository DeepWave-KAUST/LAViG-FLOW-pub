###############################################################################
# Rectified Flow Scheduler (PA-VDM / OpenSora inspired) (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Lightweight adaptation of the Rectified Flow scheduler from the PA-VDM /
#   OpenSora project (https://github.com/desaixie/pa_vdm/tree/main/opensora/rf),
#   tuned for joint CO₂ gas + pressure DiTV models. Handles progressive training
#   losses and inference timesteps with optional timestep transforms.
###############################################################################

import random

from typing import Optional

import torch
from torch.distributions import LogisticNormal
from einops import repeat, rearrange


def mean_flat(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """Mean over all non-batch dimensions with optional temporal mask."""
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    assert tensor.dim() == 5, f"Expected 5D tensor, got {tensor.shape}"
    assert tensor.shape[2] == mask.shape[1], "Mask temporal dim mismatch"
    tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
    denom = mask.sum(dim=1) * tensor.shape[-1]
    denom = torch.clamp(denom, min=1e-8)
    loss = (tensor * mask.unsqueeze(2)).sum(dim=2).sum(dim=1) / denom
    return loss


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape):
    """Extract values from 1D tensor and broadcast to desired shape."""
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1000,
):
    # Ensure precision does not drop when kwargs are fp16
    for key in ["height", "width", "num_frames"]:
        if key in model_kwargs and model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()

    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


# --------------------------------------------------
# Scheduler core
# --------------------------------------------------

class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        pa_vdm=False,
        noise_pattern="linear",
        linear_variance_scale=0.1,
        linear_shift_scale=0.3,
        latent_chunk_size=1,
        keep_x0=False,
        variable_length=False,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        assert sample_method in ["uniform", "logit-normal"]
        if use_discrete_timesteps:
            assert sample_method == "uniform", "Discrete timesteps only support uniform sampling"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        self.pa_vdm = pa_vdm
        self.noise_pattern = noise_pattern
        self.linear_variance_scale = linear_variance_scale
        self.linear_shift_scale = linear_shift_scale
        if pa_vdm:
            assert not use_discrete_timesteps, "pa_vdm not supported with discrete timesteps"

        self.training_all_progressive_timesteps = None
        self.training_num_stages = None
        self.training_latent_chunk_size = latent_chunk_size
        self.training_progressive_timesteps_stages = None
        self.keep_x0 = keep_x0
        self.variable_length = variable_length

    # -----------------
    # Training losses
    # -----------------
    def training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        noise=None,
        mask=None,
        loss_mask=None,
        weights=None,
        t=None,
        channel_last=False,
    ):
        """
        Compute RF training loss. Inputs assumed [B, C, F, H, W] unless channel_last=True.
        """
        if channel_last:
            # convert to channels-first for RF math
            x_start = x_start.permute(0, 2, 1, 3, 4)

        if self.pa_vdm and self.keep_x0:
            raise NotImplementedError("keep_x0=True not yet supported in this adaptation")

        b, c, f, h, w = x_start.shape

        if t is None:
            if self.pa_vdm:
                raise NotImplementedError("pa_vdm progressive training not yet integrated")
            else:
                if self.use_discrete_timesteps:
                    t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device)
                elif self.sample_method == "uniform":
                    t = torch.rand((b,), device=x_start.device) * self.num_timesteps
                elif self.sample_method == "logit-normal":
                    t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform and model_kwargs is not None:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        timepoints = 1.0 - t.float() / self.num_timesteps
        timepoints = timepoints.view(-1, 1, 1, 1, 1)

        x_t = timepoints * x_start + (1.0 - timepoints) * noise

        if mask is not None:
            assert mask.shape == (b, f)
            mask_broadcast = mask[:, None, :, None, None]
            x_t0 = x_start
            x_t = torch.where(mask_broadcast.bool(), x_t, x_t0)

        model_output = model(x_t, t, **model_kwargs)
        if channel_last:
            model_output = model_output.permute(0, 2, 1, 3, 4)

        velocity_pred, _ = model_output.chunk(2, dim=1)

        target = x_start - noise
        loss = (velocity_pred - target).pow(2)
        reduction_mask = loss_mask if loss_mask is not None else mask
        loss = mean_flat(loss, reduction_mask)

        if channel_last:
            return loss, t
        return loss, t

    # -----------------
    # Sampling helpers
    # -----------------
    def prepare_inference_timesteps(self, batch_size, device, model_kwargs=None):
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.full((batch_size,), t, device=device, dtype=torch.float32) for t in timesteps]
        if self.use_timestep_transform and model_kwargs is not None:
            timesteps = [timestep_transform(t, model_kwargs, num_timesteps=self.num_timesteps) for t in timesteps]
        return timesteps

    def inference_step(self, model, latents, t, dt, model_kwargs=None, channel_last=False):
        if channel_last:
            latents_cf = latents.permute(0, 2, 1, 3, 4)
        else:
            latents_cf = latents

        model_out = model(latents_cf, t, **(model_kwargs or {}))
        if channel_last:
            model_out = model_out.permute(0, 2, 1, 3, 4)

        split_dim = 2 if channel_last else 1
        velocity, _ = model_out.chunk(2, dim=split_dim)
        latents = latents + velocity * dt[:, None, None, None, None]
        return latents