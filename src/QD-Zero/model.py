"""qd-zero curve estimation and strict checkpoint loading."""

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, padding=dilation,
            dilation=dilation, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class QDZero(nn.Module):
    """Spatially preserving curve branch; implements Eq. (1).

    The denoising stage is a separately trained model, supplied at inference.
    Returns the halfway image, final enhanced image, and curve parameter maps.
    """

    def __init__(self, channels=32, iterations=8, dilations=(1, 2, 5, 1, 2, 5)):
        super().__init__()
        if channels < 1 or iterations < 1:
            raise ValueError("channels and iterations must be positive")
        if len(dilations) != 6 or any(d < 1 for d in dilations):
            raise ValueError("six positive dilation rates are required")
        self.model_config = {
            "channels": channels, "iterations": iterations,
            "dilations": list(dilations),
        }
        self.iterations = iterations
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = DepthwiseSeparableConv(3, channels, dilations[0])
        self.e_conv2 = DepthwiseSeparableConv(channels, channels, dilations[1])
        self.e_conv3 = DepthwiseSeparableConv(channels, channels, dilations[2])
        self.e_conv4 = DepthwiseSeparableConv(channels, channels, dilations[3])
        self.e_conv5 = DepthwiseSeparableConv(2 * channels, channels, dilations[4])
        self.e_conv6 = DepthwiseSeparableConv(2 * channels, channels, dilations[5])
        self.e_conv7 = nn.Conv2d(2 * channels, 3 * iterations, 1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("input must be an RGB tensor with shape [B, 3, H, W]")
        x1 = self.relu(self.e_conv1(image))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat((x3, x4), dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat((x2, x5), dim=1)))
        curves = torch.tanh(self.e_conv7(torch.cat((x1, x6), dim=1)))
        enhanced = image
        halfway = image
        for index, curve in enumerate(curves.split(3, dim=1)):
            enhanced = enhanced + curve * enhanced * (1.0 - enhanced)
            if index + 1 == max(1, self.iterations // 2):
                halfway = enhanced
        return halfway, enhanced, curves


def load_qd_zero(checkpoint_path, device="cpu"):
    """Reject incompatible architectures instead of partially loading weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "qd-zero":
        raise ValueError(
            "A trained qd-zero checkpoint containing architecture, model_config "
            "and state_dict is required. Other network weights are incompatible."
        )
    network = QDZero(**checkpoint["model_config"]).to(device)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network, checkpoint
