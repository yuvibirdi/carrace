"""The driving network: one camera frame in, the next CHUNK actions out.

Shared by train.py and drive.py so both prepare frames the same way.
"""
import io

import numpy as np
import torch
from PIL import Image
from torch import nn

W, H = 160, 120  # frames are shrunk to this before the network sees them
CHUNK = 10  # actions predicted per frame: the one for this frame and the 9 after (1/3 s at 30 fps)
LEFT, STRAIGHT, RIGHT = 0, 1, 2


def prepare(jpeg):
    """JPEG bytes -> uint8 array [H, W, 3].

    The camera's 320x240 picture is halved inside the JPEG decoder itself. The program
    on the Pi (pi/drive.cpp) asks its decoder for the same thing, so the network sees
    the same pixels in training and on the car."""
    image = Image.open(io.BytesIO(jpeg))
    image.draft("RGB", (W, H))
    image = image.convert("RGB")
    if image.size != (W, H):
        image = image.resize((W, H), Image.BILINEAR)
    return np.array(image)


def steer_class(left, right):
    return STRAIGHT if left == right else LEFT if left else RIGHT


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def conv(a, b, k, s):
            return [nn.Conv2d(a, b, k, s), nn.BatchNorm2d(b), nn.ReLU()]

        self.features = nn.Sequential(
            *conv(3, 24, 5, 2), *conv(24, 36, 5, 2), *conv(36, 48, 5, 2),
            *conv(48, 64, 3, 1), *conv(64, 64, 3, 1), nn.Flatten(),
        )
        size = self.features(torch.zeros(1, 3, H, W)).shape[1]
        self.body = nn.Sequential(nn.Dropout(0.3), nn.Linear(size, 128), nn.ReLU())
        self.steer = nn.Linear(128, CHUNK * 3)
        self.throttle = nn.Linear(128, CHUNK)

    def forward(self, frames):
        """frames: uint8 or float [N, H, W, 3] in 0..255.
        Returns steering logits [N, CHUNK, 3] and throttle logits [N, CHUNK]."""
        x = self.body(self.features(frames.permute(0, 3, 1, 2).float() / 255 - 0.5))
        return self.steer(x).view(-1, CHUNK, 3), self.throttle(x)
