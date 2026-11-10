"""The one place that picks the PyTorch device.

Order is CUDA, MPS, then CPU. Set FACEMEASURE_DEVICE to cpu, mps, cuda or cuda:N
to force a device, or to auto for the default order.
"""

import os
import re

import torch

DEVICE_ENV = "FACEMEASURE_DEVICE"
_CUDA = re.compile(r"cuda(:\d+)?")


def pick_device() -> torch.device:
    """Return the device every part of the project should use."""
    forced = os.environ.get(DEVICE_ENV, "").strip().lower()
    if forced and forced != "auto":
        return _forced_device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _forced_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"{DEVICE_ENV}=mps but MPS is not available on this machine")
        return torch.device("mps")
    if _CUDA.fullmatch(name):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{DEVICE_ENV}={name} but CUDA is not available on this machine")
        device = torch.device(name)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"{DEVICE_ENV}={name} but only {torch.cuda.device_count()} CUDA device(s) exist")
        return device
    raise ValueError(f"{DEVICE_ENV}={name!r} is not one of auto, cpu, mps, cuda, cuda:N")
