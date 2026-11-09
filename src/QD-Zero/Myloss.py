"""Reference-free perceptual constraints and quantile-dispersion loss."""

import torch
from torch import nn
from torch.nn import functional as F


class L_color(nn.Module):
    def forward(self, image):
        red, green, blue = image.mean(dim=(2, 3), keepdim=True).split(1, dim=1)
        # Preserve the fourth-power color-constancy formulation.
        return ((red - green).pow(4) + (red - blue).pow(4)
                + (blue - green).pow(4) + 1e-12).sqrt()


class L_spa(nn.Module):
    def __init__(self):
        super().__init__()
        kernels = torch.tensor([
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        ], dtype=torch.float32).unsqueeze(1)
        self.register_buffer("kernels", kernels)

    def forward(self, original, enhanced):
        original = F.avg_pool2d(original.mean(1, keepdim=True), 4)
        enhanced = F.avg_pool2d(enhanced.mean(1, keepdim=True), 4)
        difference = (F.conv2d(original, self.kernels, padding=1)
                      - F.conv2d(enhanced, self.kernels, padding=1))
        return difference.square().sum(dim=1, keepdim=True)


class L_exp(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super().__init__()
        self.patch_size = patch_size
        self.mean_val = mean_val

    def forward(self, image):
        patch = min(self.patch_size, image.shape[-2], image.shape[-1])
        local_mean = F.avg_pool2d(image.mean(1, keepdim=True), patch)
        return (local_mean - self.mean_val).square().mean()


class L_TV(nn.Module):
    def forward(self, curves):
        batch, _, height, width = curves.shape
        total = curves.new_zeros(())
        if height > 1:
            total = total + (curves[:, :, 1:] - curves[:, :, :-1]).square().sum() / (
                batch * (height - 1) * width)
        if width > 1:
            total = total + (curves[:, :, :, 1:] - curves[:, :, :, :-1]).square().sum() / (
                batch * height * (width - 1))
        return 2.0 * total


class L_QD(nn.Module):
    """Eqs. (3)-(5), computed per image, then averaged across the batch.

    L_D uses one global RGB mean per image, rather than a channel-wise mean
    or a mean shared across different images. Quartiles are per RGB channel.
    """

    def __init__(self, omega=1.0, lower_target=0.3, upper_target=0.75):
        super().__init__()
        if omega < 0 or not 0 <= lower_target <= upper_target <= 1:
            raise ValueError("invalid QD loss parameters")
        self.omega = omega
        self.lower_target = lower_target
        self.upper_target = upper_target

    def terms(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("QD loss requires [B, 3, H, W] RGB input")
        pixels = image.float().flatten(start_dim=2)
        q1 = torch.quantile(pixels, 0.25, dim=-1, interpolation="linear")
        q3 = torch.quantile(pixels, 0.75, dim=-1, interpolation="linear")
        quantile = ((q1 - self.lower_target).abs()
                    + (q3 - self.upper_target).abs()).mean()
        global_mean = pixels.mean(dim=(1, 2), keepdim=True)
        dispersion = (pixels - global_mean).abs().mean()
        return quantile, dispersion

    def forward(self, image):
        quantile, dispersion = self.terms(image)
        return quantile + self.omega * dispersion


class QDZeroLoss(nn.Module):
    """Eq. (2). Defaults are configurable implementation settings."""

    def __init__(self, lambda_spa=1.0, lambda_exp=10.0, lambda_col=5.0,
                 lambda_tv=200.0, lambda_qd=1.0, omega=1.0):
        super().__init__()
        self.spatial = L_spa()
        self.exposure = L_exp()
        self.color = L_color()
        self.tv = L_TV()
        self.qd = L_QD(omega=omega)
        self.weights = dict(spa=lambda_spa, exp=lambda_exp, col=lambda_col,
                            tv=lambda_tv, qd=lambda_qd)
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("loss weights must be nonnegative")

    def forward(self, original, enhanced, curves):
        quantile, dispersion = self.qd.terms(enhanced)
        terms = {
            "spa": self.spatial(original, enhanced).mean(),
            "exp": self.exposure(enhanced),
            "col": self.color(enhanced).mean(),
            "tv": self.tv(curves),
            "qd": quantile + self.qd.omega * dispersion,
        }
        total = sum(self.weights[key] * value for key, value in terms.items())
        terms.update(quantile=quantile, dispersion=dispersion)
        return total, terms
