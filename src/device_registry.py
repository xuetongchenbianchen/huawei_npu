"""Device name to backend module registry for benchmark runners."""
from __future__ import annotations


DEVICE_MODULES = {
    "cpu": "src.cpu",
    "cuda": "src.cuda",
    "npu": "src.npu",
}


def normalize_device(device: str | None) -> str:
    return (device or "").strip().lower()


def module_for_device(device: str) -> str:
    normalized = normalize_device(device)
    if normalized not in DEVICE_MODULES:
        known = ", ".join(sorted(DEVICE_MODULES))
        raise ValueError(f"未知 device: {device!r}。已知 device: {known}")
    return DEVICE_MODULES[normalized]


def resolve_backend(device: str, module_override: str | None = None) -> tuple[str, str]:
    normalized = normalize_device(device)
    return normalized, module_override or module_for_device(normalized)
