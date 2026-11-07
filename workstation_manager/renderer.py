"""Render a CollectionResult as human-readable text or JSON.

All None values are surfaced explicitly — renderers never substitute zero or
hide gaps.  Human output labels None as 'unavailable'; JSON uses null.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .snapshot import CollectionResult, DeviceSnapshot, ProcessInfo


def _bytes_to_mib(b: Optional[int]) -> Optional[float]:
    if b is None:
        return None
    return round(b / (1024 * 1024), 1)


def _fmt_mib(b: Optional[int]) -> str:
    mib = _bytes_to_mib(b)
    if mib is None:
        return "unavailable"
    return f"{mib} MiB"


def _fmt_percent(v: Optional[int]) -> str:
    if v is None:
        return "unavailable"
    return f"{v}%"


def _fmt_comm(comm: Optional[str]) -> str:
    if comm is None:
        return "(name unavailable)"
    return comm


# ---------------------------------------------------------------------------
# Human-readable
# ---------------------------------------------------------------------------

def render_text(result: CollectionResult) -> str:
    lines: List[str] = []

    if result.budget_exhausted:
        lines.append("WARNING: collection budget exhausted — output is partial")

    for note in result.notes:
        lines.append(f"NOTE: {note}")

    if not result.devices:
        lines.append("No AMD GPU devices found.")
        return "\n".join(lines)

    for dev in result.devices:
        lines.append("")
        lines.append(f"Device: {dev.label}")
        lines.append(f"  Path:       {dev.path}")
        lines.append(f"  PCI ID:     {dev.pci_id}")
        lines.append(f"  VRAM total: {_fmt_mib(dev.vram_total_bytes)}")
        lines.append(f"  VRAM used:  {_fmt_mib(dev.vram_used_bytes)}")
        lines.append(f"  VRAM free:  {_fmt_mib(dev.vram_free_bytes)}")
        lines.append(f"  GPU busy:   {_fmt_percent(dev.gpu_busy_percent)}")

        if dev.processes:
            lines.append(f"  Processes ({len(dev.processes)} observable DRM fd holders):")
            for p in dev.processes:
                lines.append(f"    PID {p.pid}: {_fmt_comm(p.comm)}")
        else:
            lines.append("  Processes: none observable (or fdinfo unreadable)")

        for note in dev.notes:
            lines.append(f"  NOTE: {note}")

    lines.append("")
    lines.append(
        "NOTE: Observed DRM handles identify device users, not proof of active"
        " computation or authorization to launch work."
    )
    lines.append(
        "NOTE: Limited process visibility is not evidence the GPU is idle."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _process_to_dict(p: ProcessInfo) -> Dict[str, Any]:
    return {"pid": p.pid, "comm": p.comm}


def _device_to_dict(dev: DeviceSnapshot) -> Dict[str, Any]:
    return {
        "path": dev.path,
        "label": dev.label,
        "pci_id": dev.pci_id,
        "vram_total_bytes": dev.vram_total_bytes,
        "vram_used_bytes": dev.vram_used_bytes,
        "vram_free_bytes": dev.vram_free_bytes,
        "gpu_busy_percent": dev.gpu_busy_percent,
        "processes": [_process_to_dict(p) for p in dev.processes],
        "notes": list(dev.notes),
    }


def render_json(result: CollectionResult) -> str:
    data: Dict[str, Any] = {
        "budget_exhausted": result.budget_exhausted,
        "notes": list(result.notes),
        "devices": [_device_to_dict(d) for d in result.devices],
    }
    return json.dumps(data, indent=2)
