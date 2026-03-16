from __future__ import annotations

from typing import Any


AVAILABLE_BASELINE_METHODS: tuple[str, ...] = ("reno", "bbr", "cubic")
DEFAULT_BASELINE_METHODS: tuple[str, ...] = AVAILABLE_BASELINE_METHODS
LEGACY_BASELINE_METHODS: tuple[str, ...] = ("cubic", "bbr")
BASELINE_CONTROLLER_SPECS: dict[str, tuple[str, str]] = {
    "reno": ("reno", "kernel_cc"),
    "bbr": ("bbr", "kernel_cc"),
    "cubic": ("cubic", "kernel_cc"),
}
BASELINE_LABELS: dict[str, str] = {
    "reno": "Reno",
    "bbr": "BBR",
    "cubic": "CUBIC",
}


def normalize_baseline_methods(
    value: Any,
    *,
    default: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    if value is None:
        methods = tuple(default if default is not None else DEFAULT_BASELINE_METHODS)
    elif isinstance(value, str):
        methods = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    else:
        methods = tuple(str(part).strip().lower() for part in value if str(part).strip())
    if not methods:
        raise ValueError("at least one baseline method must be enabled")
    normalized: list[str] = []
    seen: set[str] = set()
    for method in methods:
        if method not in AVAILABLE_BASELINE_METHODS:
            raise ValueError(
                f"unsupported baseline method '{method}'; available: {', '.join(AVAILABLE_BASELINE_METHODS)}"
            )
        if method in seen:
            continue
        normalized.append(method)
        seen.add(method)
    return tuple(normalized)


def baseline_methods_from_config(config_payload: dict[str, Any]) -> tuple[str, ...]:
    if "baseline_methods" not in config_payload:
        return LEGACY_BASELINE_METHODS
    return normalize_baseline_methods(config_payload.get("baseline_methods"), default=DEFAULT_BASELINE_METHODS)
