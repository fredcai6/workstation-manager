"""Immutable snapshot data types.

Every numeric field uses None to signal unavailable or invalid readings so
callers can distinguish:
  - measured value   (int, including zero)
  - unavailable      (None — file absent or permission denied)
  - invalid          (None — content unparseable or out of range)
  - partial          (struct returned but some sub-fields are None)

Renderers must surface None explicitly; they must not substitute zero or hide
the gap.  Callers must test for None before arithmetic.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class ProcessInfo:
    """Best-effort attribution of one DRM file-descriptor holder."""

    pid: int
    # None means /proc/<pid>/comm was unreadable (vanished or EPERM).
    comm: Optional[str]


@dataclasses.dataclass(frozen=True)
class DeviceSnapshot:
    """Snapshot of one logical GPU, deduplicated by PCI identity."""

    # Canonical sysfs path for the chosen alias, e.g. /sys/class/drm/card0.
    path: str

    # Human label built from sysfs (model name when available, else path).
    label: str

    # PCI address used for alias deduplication, e.g. "0000:03:00.0".
    # "unknown" when undetermined.
    pci_id: str

    # VRAM in bytes; None when unavailable or invalid.
    vram_total_bytes: Optional[int]
    vram_used_bytes: Optional[int]

    # Derived only when both total and used are valid integers and used <= total.
    # None otherwise — two reads in one pass are not atomic.
    vram_free_bytes: Optional[int]

    # GPU busy 0-100; None when unavailable or invalid.
    # Values outside 0..100 are set to None (invalid, not clamped).
    gpu_busy_percent: Optional[int]

    # Observable DRM fd holders; empty list when fdinfo is unreadable.
    processes: Sequence[ProcessInfo] = dataclasses.field(default_factory=list)

    # Human-readable notes about gaps or anomalies observed during collection.
    notes: Sequence[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class CollectionResult:
    """Top-level result of one collection pass."""

    devices: Sequence[DeviceSnapshot]

    # True when the cooperative budget was exhausted before scanning completed.
    budget_exhausted: bool = False

    # Notes not tied to a specific device.
    notes: Sequence[str] = dataclasses.field(default_factory=list)
