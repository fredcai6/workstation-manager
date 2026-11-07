"""AMD GPU collector — reads sysfs/DRM only, no writes, stdlib only.

Responsibilities:
  - Enumerate /sys/class/drm card* entries, filter to AMD, deduplicate aliases
    by PCI identity (uevent PCI_SLOT_NAME or realpath component).
  - Read VRAM total/used bytes and GPU busy percent from hwmon/sysfs.
  - Enumerate /proc/<pid>/fdinfo to find DRM fd holders for each device.
  - Read /proc/<pid>/comm for process names; never read environ or argv.
  - Respect a cooperative monotonic budget throughout; return partial results
    on exhaustion.  Cannot guarantee individual kernel reads won't block.
  - Return None for unavailable/invalid metrics; never clamp or invent values.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from .snapshot import CollectionResult, DeviceSnapshot, ProcessInfo

_DEFAULT_DRM_ROOT = "/sys/class/drm"
_DEFAULT_PROC_ROOT = "/proc"
_DEFAULT_BUDGET_SECONDS = 5.0

# AMD PCI vendor ID
_AMD_VENDOR = "0x1002"


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------

def _read_text(path: str) -> Optional[str]:
    """Return stripped file text, or None on any OSError."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _parse_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_int_from_file(path: str) -> Tuple[Optional[int], str]:
    """Read path and parse as int.  Returns (value_or_None, note_string)."""
    raw = _read_text(path)
    if raw is None:
        return None, "unavailable"
    v = _parse_int(raw)
    if v is None:
        return None, f"invalid({raw!r})"
    return v, ""


# ---------------------------------------------------------------------------
# Device discovery helpers
# ---------------------------------------------------------------------------

def _pci_id_for_entry(entry_path: str) -> str:
    """Return PCI slot name for a DRM entry, or 'unknown'."""
    dev = os.path.join(entry_path, "device")
    raw = _read_text(os.path.join(dev, "uevent"))
    if raw:
        for line in raw.splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    # Fallback: find PCI-address-shaped component in realpath.
    try:
        real = os.path.realpath(entry_path)
        for part in reversed(real.split(os.sep)):
            if len(part) >= 7 and part.count(":") >= 1 and "." in part:
                return part
    except OSError:
        pass
    return "unknown"


def _is_amd(entry_path: str) -> bool:
    """Return True when sysfs identifies this as an AMD GPU."""
    dev = os.path.join(entry_path, "device")
    vendor = _read_text(os.path.join(dev, "vendor"))
    if vendor is not None:
        return vendor.lower() in ("0x1002", "1002")
    uevent = _read_text(os.path.join(dev, "uevent"))
    if uevent:
        for line in uevent.splitlines():
            if line.startswith("PCI_ID=") and "1002:" in line.upper():
                return True
    return False


def _device_label(entry_path: str) -> str:
    """Build a human label from sysfs product_name or fall back to path."""
    dev = os.path.join(entry_path, "device")
    name = _read_text(os.path.join(dev, "product_name"))
    if name:
        return name
    # Try hwmon label
    for candidate in ("product_name", "name"):
        v = _read_text(os.path.join(dev, candidate))
        if v:
            return v
    return os.path.basename(entry_path)


# ---------------------------------------------------------------------------
# VRAM and busy metrics
# ---------------------------------------------------------------------------

def _read_vram(entry_path: str) -> Tuple[Optional[int], Optional[int], List[str]]:
    """Return (total_bytes, used_bytes, notes) from AMDGPU sysfs."""
    notes: List[str] = []
    base = os.path.join(entry_path, "device", "mem_info_vram_total")
    total, note = _parse_int_from_file(base)
    if note:
        notes.append(f"vram_total: {note}")
    used_path = os.path.join(entry_path, "device", "mem_info_vram_used")
    used, note = _parse_int_from_file(used_path)
    if note:
        notes.append(f"vram_used: {note}")
    return total, used, notes


def _vram_free(total: Optional[int], used: Optional[int]) -> Optional[int]:
    """Compute free VRAM only when both values are valid and used <= total."""
    if total is None or used is None:
        return None
    if used < 0 or total < 0 or used > total:
        return None
    return total - used


def _read_busy(entry_path: str) -> Tuple[Optional[int], List[str]]:
    """Return (busy_percent, notes) from gpu_busy_percent sysfs file."""
    notes: List[str] = []
    path = os.path.join(entry_path, "device", "gpu_busy_percent")
    v, note = _parse_int_from_file(path)
    if note:
        notes.append(f"gpu_busy_percent: {note}")
        return None, notes
    if v is not None and not (0 <= v <= 100):
        notes.append(f"gpu_busy_percent: invalid({v!r} outside 0-100)")
        return None, notes
    return v, notes


# ---------------------------------------------------------------------------
# Process attribution via DRM fdinfo
# ---------------------------------------------------------------------------

def _drm_minor_for_entry(entry_path: str) -> Optional[str]:
    """Return the DRM minor node name (e.g. 'renderD128') for this card entry."""
    # The render node is a sibling of the card entry; look for renderD* under device/drm/.
    drm_dir = os.path.join(entry_path, "device", "drm")
    try:
        for name in os.listdir(drm_dir):
            if name.startswith("renderD"):
                return name
    except OSError:
        pass
    # Fallback: card -> renderD mapping by card number parity is not reliable.
    return None


def _normalise_pci(addr: str) -> str:
    """Lower-case a PCI address for case-insensitive comparison."""
    return addr.strip().lower()


def _pids_using_device(
    entry_path: str,
    proc_root: str,
    deadline: float,
    pci_id: str = "",
) -> Tuple[List[ProcessInfo], List[str]]:
    """Scan /proc/<pid>/fdinfo for DRM fd holders referencing this device.

    Device matching uses the canonical DRM fdinfo key ``drm-pdev`` (PCI
    address), compared case-insensitively to the device's ``pci_id``.  If the
    fdinfo file has no ``drm-pdev`` line we fall back to checking for the
    render-node name as a substring (best-effort for older kernels).

    Unreadable and disappearing fdinfo entries are counted as visibility gaps
    and recorded as device notes.

    This is best-effort: processes may vanish, files may be unreadable, and
    the budget may expire.
    """
    procs: List[ProcessInfo] = []
    notes: List[str] = []
    seen_pids: Set[int] = set()
    directory_denied = directory_vanished = directory_unavailable = 0
    file_vanished = file_unavailable = 0
    norm_pci = _normalise_pci(pci_id) if pci_id else ""

    render_name = _drm_minor_for_entry(entry_path)

    try:
        proc_entries = os.listdir(proc_root)
    except OSError as exc:
        notes.append(f"proc scan unavailable: {exc}")
        return procs, notes

    for entry in proc_entries:
        if time.monotonic() >= deadline:
            notes.append("budget exhausted during process scan")
            break
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in seen_pids:
            continue

        fdinfo_dir = os.path.join(proc_root, entry, "fdinfo")
        try:
            fd_entries = os.listdir(fdinfo_dir)
        except PermissionError:
            directory_denied += 1
            continue
        except FileNotFoundError:
            directory_vanished += 1
            continue
        except OSError:
            directory_unavailable += 1
            continue

        found = False
        perm_denied_count = 0
        for fd_name in fd_entries:
            if found:
                break
            if time.monotonic() >= deadline:
                notes.append("budget exhausted during fdinfo scan")
                break
            fdinfo_path = os.path.join(fdinfo_dir, fd_name)
            # Read with explicit error distinction.
            try:
                with open(fdinfo_path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except PermissionError:
                perm_denied_count += 1
                continue
            except FileNotFoundError:
                file_vanished += 1
                continue
            except OSError:
                file_unavailable += 1
                continue

            # Parse fdinfo keys.
            is_amdgpu = False
            drm_pdev: Optional[str] = None
            refs_render = False
            for line in content.splitlines():
                low = line.lower()
                if low.startswith("drm-driver:") and "amdgpu" in low:
                    is_amdgpu = True
                if low.startswith("drm-pdev:"):
                    drm_pdev = line.split(":", 1)[1].strip()
                if render_name and render_name in line:
                    refs_render = True

            if not is_amdgpu:
                continue

            # Primary match: drm-pdev == our pci_id.
            if drm_pdev is not None:
                if norm_pci and _normalise_pci(drm_pdev) == norm_pci:
                    found = True
            elif refs_render:
                # Fallback for kernels that don't emit drm-pdev.
                found = True

        if perm_denied_count:
            notes.append(
                f"partial fdinfo visibility: {perm_denied_count} file(s)"
                f" unreadable for pid {pid}"
            )

        if found:
            seen_pids.add(pid)
            comm_path = os.path.join(proc_root, entry, "comm")
            comm = _read_text(comm_path)  # None if vanished/unreadable
            procs.append(ProcessInfo(pid=pid, comm=comm))

    for count, detail in (
        (directory_denied, "fdinfo directories unreadable (permission denied)"),
        (directory_vanished, "fdinfo directories disappeared during scan"),
        (directory_unavailable, "fdinfo directories unavailable"),
        (file_vanished, "fdinfo files disappeared during scan"),
        (file_unavailable, "fdinfo files unavailable"),
    ):
        if count:
            notes.append(f"partial process visibility: {count} {detail}")
    return procs, notes


# ---------------------------------------------------------------------------
# Budget helper
# ---------------------------------------------------------------------------

class _Budget:
    """Cooperative monotonic deadline tracker."""

    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds

    @property
    def deadline(self) -> float:
        return self._deadline

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def exhausted(self) -> bool:
        return time.monotonic() >= self._deadline


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect(
    drm_root: str = _DEFAULT_DRM_ROOT,
    proc_root: str = _DEFAULT_PROC_ROOT,
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
) -> CollectionResult:
    """Collect a snapshot of all AMD GPU devices.

    Parameters
    ----------
    drm_root:
        Directory to enumerate for DRM card entries (default /sys/class/drm).
    proc_root:
        /proc root for process scanning (default /proc).
    budget_seconds:
        Cooperative time budget in seconds (default 5.0).  Must be a finite
        positive number.  The collector checks the budget between operations;
        individual kernel reads may exceed it.

    Returns
    -------
    CollectionResult with all discovered devices and any collection notes.

    Raises
    ------
    ValueError
        If budget_seconds is not a finite positive number.
    """
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError(
            f"budget_seconds must be a finite positive number, got {budget_seconds!r}"
        )

    budget = _Budget(budget_seconds)
    top_notes: List[str] = []
    budget_exhausted = False

    # Enumerate DRM entries.
    try:
        all_entries = sorted(os.listdir(drm_root))
    except OSError as exc:
        top_notes.append(f"drm_root unavailable ({drm_root}): {exc}")
        return CollectionResult(devices=[], budget_exhausted=False, notes=top_notes)

    # Filter to card* entries (not render nodes or connectors).
    card_entries = [e for e in all_entries if e.startswith("card")]

    # Discover AMD devices; deduplicate by PCI id.
    # dedup_key -> canonical entry path (first seen wins).
    # For unknown PCI IDs we use a unique per-path key so they're never merged.
    seen_dedup: Dict[str, str] = {}
    amd_cards: List[Tuple[str, str]] = []  # (entry_path, pci_id)

    for name in card_entries:
        if budget.exhausted():
            budget_exhausted = True
            top_notes.append("budget exhausted during device discovery")
            break
        entry_path = os.path.join(drm_root, name)
        if not _is_amd(entry_path):
            continue
        pci_id = _pci_id_for_entry(entry_path)
        # Devices with unknown PCI ID get a unique key so they're never merged.
        dedup_key = pci_id if pci_id != "unknown" else f"unknown:{entry_path}"
        if dedup_key in seen_dedup:
            # Alias of an already-seen device; skip.
            continue
        seen_dedup[dedup_key] = entry_path
        amd_cards.append((entry_path, pci_id))

    # ── Pass 1: read VRAM + busy for every discovered device ────────────────
    # We do this before process scanning so that even if process scanning
    # exhausts the budget, every device's metric fields are populated.

    # Each entry: (entry_path, pci_id, label, vram_total, vram_used, busy, notes_list)
    pass1: List[Tuple[str, str, str, Optional[int], Optional[int], Optional[int], List[str]]] = []

    for entry_path, pci_id in amd_cards:
        if budget.exhausted():
            budget_exhausted = True
            top_notes.append("budget exhausted during metric collection")
            # Still record all remaining devices with None metrics so they appear.
            for rem_path, rem_pci in amd_cards[len(pass1):]:
                rem_label = _device_label(rem_path)
                pass1.append((rem_path, rem_pci, rem_label, None, None, None,
                               ["budget exhausted before metrics read"]))
            break

        dev_notes: List[str] = []
        label = _device_label(entry_path)

        vram_total, vram_used, vram_notes = _read_vram(entry_path)
        dev_notes.extend(vram_notes)

        # Validate negative raw values.
        if vram_total is not None and vram_total < 0:
            dev_notes.append("vram_total: invalid(negative)")
            vram_total = None
        if vram_used is not None and vram_used < 0:
            dev_notes.append("vram_used: invalid(negative)")
            vram_used = None
        if (vram_total is not None and vram_used is not None
                and vram_used > vram_total):
            dev_notes.append("vram_free: inconsistent (used > total)")

        busy, busy_notes = _read_busy(entry_path)
        dev_notes.extend(busy_notes)

        pass1.append((entry_path, pci_id, label, vram_total, vram_used, busy, dev_notes))

    # ── Pass 2: process scanning, one device at a time ───────────────────────
    devices: List[DeviceSnapshot] = []

    for entry_path, pci_id, label, vram_total, vram_used, busy, dev_notes in pass1:
        notes = list(dev_notes)  # copy so we can append proc notes

        if budget.exhausted():
            budget_exhausted = True
            notes.append("budget exhausted before process scan")
            devices.append(DeviceSnapshot(
                path=entry_path,
                label=label,
                pci_id=pci_id,
                vram_total_bytes=vram_total,
                vram_used_bytes=vram_used,
                vram_free_bytes=_vram_free(vram_total, vram_used),
                gpu_busy_percent=busy,
                processes=[],
                notes=notes,
            ))
            continue

        procs, proc_notes = _pids_using_device(
            entry_path, proc_root, budget.deadline, pci_id=pci_id
        )
        notes.extend(proc_notes)
        if any("budget exhausted" in n for n in proc_notes):
            budget_exhausted = True

        devices.append(DeviceSnapshot(
            path=entry_path,
            label=label,
            pci_id=pci_id,
            vram_total_bytes=vram_total,
            vram_used_bytes=vram_used,
            vram_free_bytes=_vram_free(vram_total, vram_used),
            gpu_busy_percent=busy,
            processes=procs,
            notes=notes,
        ))

    return CollectionResult(
        devices=devices,
        budget_exhausted=budget_exhausted,
        notes=top_notes,
    )
