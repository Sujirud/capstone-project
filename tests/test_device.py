"""facemeasure.device.pick_device with CUDA and MPS availability faked."""

import pytest
import torch

from facemeasure import device as dev


@pytest.fixture
def fake_hw(monkeypatch):
    """Set which accelerators look available, with no FACEMEASURE_DEVICE."""
    monkeypatch.delenv(dev.DEVICE_ENV, raising=False)

    def set_hw(cuda: bool, mps: bool, cuda_count: int = 1):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: cuda_count if cuda else 0)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    return set_hw


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_order_cuda_mps_cpu(fake_hw, cuda, mps, expected):
    fake_hw(cuda=cuda, mps=mps)
    assert dev.pick_device().type == expected


@pytest.mark.parametrize("value", ["cpu", " CPU "])
def test_env_forces_cpu(fake_hw, monkeypatch, value):
    fake_hw(cuda=True, mps=True)
    monkeypatch.setenv(dev.DEVICE_ENV, value)
    assert dev.pick_device() == torch.device("cpu")


def test_env_auto_uses_default_order(fake_hw, monkeypatch):
    fake_hw(cuda=False, mps=True)
    monkeypatch.setenv(dev.DEVICE_ENV, "auto")
    assert dev.pick_device().type == "mps"


def test_env_forces_cuda_index(fake_hw, monkeypatch):
    fake_hw(cuda=True, mps=False, cuda_count=2)
    monkeypatch.setenv(dev.DEVICE_ENV, "cuda:1")
    assert dev.pick_device() == torch.device("cuda:1")


@pytest.mark.parametrize(
    ("value", "cuda", "mps"),
    [("mps", True, False), ("cuda", False, True), ("cuda:2", True, True)],
)
def test_env_unavailable_device_fails(fake_hw, monkeypatch, value, cuda, mps):
    fake_hw(cuda=cuda, mps=mps, cuda_count=1)
    monkeypatch.setenv(dev.DEVICE_ENV, value)
    with pytest.raises(RuntimeError, match=dev.DEVICE_ENV):
        dev.pick_device()


def test_env_unknown_value_fails(fake_hw, monkeypatch):
    fake_hw(cuda=False, mps=False)
    monkeypatch.setenv(dev.DEVICE_ENV, "tpu")
    with pytest.raises(ValueError, match="tpu"):
        dev.pick_device()
