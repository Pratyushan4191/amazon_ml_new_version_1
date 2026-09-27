"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: Memory & Progress Monitoring

IMPORTANT (per root-cause report): summed RSS across parent + worker
processes double-counts physically shared (copy-on-write) memory and must
NEVER be treated as "actual physical memory used". This module prefers:

  1. PSS (Proportional Set Size) from /proc/<pid>/smaps_rollup   [most accurate]
  2. cgroup memory.current                                        [container/AWS cgroup v2]
  3. psutil RSS as supplementary, single-process-only information  [never summed across processes]

All functions degrade gracefully (return None / best-effort) on platforms
where a given source isn't available (e.g. macOS has no /proc).
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import psutil


# ─────────────────────────────────────────────────────────────
# LOW-LEVEL MEMORY READERS
# ─────────────────────────────────────────────────────────────

def read_pss_bytes(pid: Optional[int] = None) -> Optional[int]:
    """
    Read PSS (Proportional Set Size) for a process from
    /proc/<pid>/smaps_rollup (Linux 4.14+). Falls back to summing the
    'Pss:' lines in /proc/<pid>/smaps if smaps_rollup is unavailable.

    PSS correctly attributes shared (fork/COW) pages proportionally, so
    summing PSS across processes is a *valid* approximation of true
    physical memory footprint -- unlike summing RSS.
    """
    pid = pid or os.getpid()
    rollup = Path(f"/proc/{pid}/smaps_rollup")
    try:
        if rollup.exists():
            for line in rollup.read_text().splitlines():
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
        smaps = Path(f"/proc/{pid}/smaps")
        if smaps.exists():
            total_kb = 0
            for line in smaps.read_text().splitlines():
                if line.startswith("Pss:"):
                    total_kb += int(line.split()[1])
            return total_kb * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def read_cgroup_memory_current() -> Optional[int]:
    """
    Read cgroup memory.current (cgroup v2) or memory.usage_in_bytes (v1).
    This is the number the kernel will actually OOM-kill against, so it's
    the most production-relevant figure on a containerized/cgrouped AWS host.
    """
    candidates = [
        Path("/sys/fs/cgroup/memory.current"),                       # cgroup v2
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),         # cgroup v1
    ]
    for p in candidates:
        try:
            if p.exists():
                return int(p.read_text().strip())
        except (PermissionError, ValueError):
            continue
    return None


def read_cgroup_memory_max() -> Optional[int]:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for p in candidates:
        try:
            if p.exists():
                txt = p.read_text().strip()
                if txt == "max":
                    return None
                val = int(txt)
                # cgroup v1 unset limit is often a huge sentinel; ignore if absurd
                if val > 10 ** 15:
                    return None
                return val
        except (PermissionError, ValueError):
            continue
    return None


def read_rss_bytes(pid: Optional[int] = None) -> Optional[int]:
    """Single-process RSS via psutil. NEVER sum this across processes."""
    try:
        p = psutil.Process(pid or os.getpid())
        return p.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def system_available_bytes() -> Optional[int]:
    try:
        return psutil.virtual_memory().available
    except Exception:
        return None


def system_total_bytes() -> Optional[int]:
    try:
        return psutil.virtual_memory().total
    except Exception:
        return None


def child_pids_pss(parent_pid: Optional[int] = None) -> Dict[int, int]:
    """
    PSS of every live child process of parent_pid (does NOT include parent).
    Summing PSS (unlike RSS) across parent + children is a defensible
    approximation of total physical footprint, because PSS already divides
    shared pages by the number of processes mapping them.
    """
    parent_pid = parent_pid or os.getpid()
    result = {}
    try:
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            pss = read_pss_bytes(child.pid)
            if pss is not None:
                result[child.pid] = pss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return result


# ─────────────────────────────────────────────────────────────
# SNAPSHOT
# ─────────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    timestamp: float
    label: str
    parent_rss: Optional[int]
    parent_pss: Optional[int]
    children_pss_total: Optional[int]
    total_pss: Optional[int]          # parent_pss + children_pss_total, when both known
    cgroup_current: Optional[int]
    cgroup_max: Optional[int]
    system_available: Optional[int]
    system_total: Optional[int]
    n_children: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "parent_rss_mb": _mb(self.parent_rss),
            "parent_pss_mb": _mb(self.parent_pss),
            "children_pss_total_mb": _mb(self.children_pss_total),
            "total_pss_mb": _mb(self.total_pss),
            "cgroup_current_mb": _mb(self.cgroup_current),
            "cgroup_max_mb": _mb(self.cgroup_max),
            "system_available_mb": _mb(self.system_available),
            "system_total_mb": _mb(self.system_total),
            "n_children": self.n_children,
        }

    def line(self) -> str:
        best = self.cgroup_current if self.cgroup_current is not None else self.total_pss
        if best is None:
            best = self.parent_rss
            best_label = "RSS"
        else:
            best_label = "cgroup" if self.cgroup_current is not None else "PSS"
        best_mb = _mb(best)
        best_str = f"{best_mb:.0f}MB" if best_mb is not None else "n/a"
        parent_rss_mb = _mb(self.parent_rss)
        parent_rss_str = f"{parent_rss_mb:.0f}MB" if parent_rss_mb is not None else "n/a"
        avail = _mb(self.system_available)
        avail_str = f"{avail:.0f}MB" if avail is not None else "n/a"
        return (
            f"[mem:{self.label}] {best_label}={best_str} "
            f"parent_rss={parent_rss_str} "
            f"children={self.n_children} "
            f"avail={avail_str}"
        )


def _mb(x: Optional[int]) -> Optional[float]:
    return None if x is None else x / (1024 * 1024)


def take_snapshot(label: str) -> MemorySnapshot:
    parent_rss = read_rss_bytes()
    parent_pss = read_pss_bytes()
    children = child_pids_pss()
    children_total = sum(children.values()) if children else (0 if parent_pss is not None else None)
    total_pss = None
    if parent_pss is not None and children_total is not None:
        total_pss = parent_pss + children_total
    return MemorySnapshot(
        timestamp=time.time(),
        label=label,
        parent_rss=parent_rss,
        parent_pss=parent_pss,
        children_pss_total=children_total,
        total_pss=total_pss,
        cgroup_current=read_cgroup_memory_current(),
        cgroup_max=read_cgroup_memory_max(),
        system_available=system_available_bytes(),
        system_total=system_total_bytes(),
        n_children=len(children),
    )


# ─────────────────────────────────────────────────────────────
# MEMORY GUARD
# ─────────────────────────────────────────────────────────────

class MemoryBudgetExceeded(RuntimeError):
    """Raised by MemoryGuard at the 'emergency' threshold so the caller can
    fail gracefully (checkpoint + exit) instead of waiting for the Linux OOM killer."""


class MemoryGuard:
    """
    Configurable warning / critical / emergency thresholds against a memory
    budget (in GB). Uses cgroup memory.current when available (most
    accurate for what will actually trigger an OOM kill), else total PSS,
    else system-wide used memory as a last resort.

    warning_frac / critical_frac / emergency_frac are fractions of
    memory_budget_gb (e.g. 0.70 / 0.85 / 0.95).
    """

    def __init__(
        self,
        memory_budget_gb: float,
        warning_frac: float = 0.70,
        critical_frac: float = 0.85,
        emergency_frac: float = 0.95,
        logger=print,
    ):
        self.budget_bytes = memory_budget_gb * (1024 ** 3)
        self.warning_bytes = self.budget_bytes * warning_frac
        self.critical_bytes = self.budget_bytes * critical_frac
        self.emergency_bytes = self.budget_bytes * emergency_frac
        self._warned = False
        self._critical_hit = False
        self.logger = logger
        self.peak_bytes = 0

    def current_usage_bytes(self, snap: Optional[MemorySnapshot] = None) -> Optional[int]:
        snap = snap or take_snapshot("guard-check")
        if snap.cgroup_current is not None:
            return snap.cgroup_current
        if snap.total_pss is not None:
            return snap.total_pss
        if snap.system_total is not None and snap.system_available is not None:
            return snap.system_total - snap.system_available
        return None

    def check(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Returns one of: "ok", "warning", "critical". Raises
        MemoryBudgetExceeded at the emergency threshold so the caller can
        checkpoint and fail gracefully rather than being killed by the OS.
        """
        snap = take_snapshot("guard-check")
        used = self.current_usage_bytes(snap)
        if used is None:
            return "ok"  # cannot measure on this platform; don't block work
        self.peak_bytes = max(self.peak_bytes, used)

        if used >= self.emergency_bytes:
            ctx = context or {}
            self.logger(
                f"  🛑 EMERGENCY: memory {used/1e9:.2f} GB >= "
                f"{self.emergency_bytes/1e9:.2f} GB (budget {self.budget_bytes/1e9:.2f} GB). "
                f"Context: {json.dumps(ctx, default=str)}"
            )
            raise MemoryBudgetExceeded(
                f"Memory {used/1e9:.2f} GB exceeded emergency threshold "
                f"{self.emergency_bytes/1e9:.2f} GB. Context: {ctx}"
            )
        if used >= self.critical_bytes:
            if not self._critical_hit:
                self.logger(
                    f"  ⚠️  CRITICAL: memory {used/1e9:.2f} GB >= "
                    f"{self.critical_bytes/1e9:.2f} GB — stop starting new work if possible."
                )
                self._critical_hit = True
            return "critical"
        if used >= self.warning_bytes:
            if not self._warned:
                self.logger(f"  ⚠️  WARNING: memory {used/1e9:.2f} GB >= {self.warning_bytes/1e9:.2f} GB")
                self._warned = True
            return "warning"
        self._warned = False
        self._critical_hit = False
        return "ok"


# ─────────────────────────────────────────────────────────────
# PROGRESS / ETA TRACKER
# ─────────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Tracks real (not fabricated) throughput and ETA. If total is unknown or
    zero, percentage/ETA are omitted rather than guessed, per the
    requirement to never fabricate progress totals.
    """

    def __init__(self, total: Optional[int], label: str = ""):
        self.total = total if (total and total > 0) else None
        self.label = label
        self.start_time = time.time()
        self.processed = 0
        self._last_report = 0.0

    def update(self, n: int = 1) -> None:
        self.processed += n

    def rate_per_sec(self) -> float:
        elapsed = max(time.time() - self.start_time, 1e-9)
        return self.processed / elapsed

    def eta_seconds(self) -> Optional[float]:
        if self.total is None:
            return None
        rate = self.rate_per_sec()
        if rate <= 0:
            return None
        remaining = max(self.total - self.processed, 0)
        return remaining / rate

    def snapshot_line(self, extra: Optional[Dict[str, Any]] = None) -> str:
        elapsed = time.time() - self.start_time
        rate = self.rate_per_sec()
        if self.total:
            parts = [f"{self.label}: {self.processed:,} / {self.total:,}"]
        else:
            parts = [f"{self.label}: {self.processed:,}"]
        if self.total:
            pct = 100.0 * self.processed / self.total
            parts.append(f"{pct:5.1f}%")
        parts.append(f"rate={rate:,.0f}/s")
        parts.append(f"elapsed={_fmt_secs(elapsed)}")
        eta = self.eta_seconds()
        if eta is not None:
            parts.append(f"ETA={_fmt_secs(eta)}")
        if extra:
            for k, v in extra.items():
                parts.append(f"{k}={v}")
        return " | ".join(parts)


def _fmt_secs(s: float) -> str:
    s = int(max(s, 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m:02d}m{sec:02d}s"


# ─────────────────────────────────────────────────────────────
# EVENT LOG (for the required "before/after each stage" telemetry)
# ─────────────────────────────────────────────────────────────

class MemoryEventLog:
    """Accumulates labeled MemorySnapshots and can dump them to JSON for the
    memory-optimization report / troubleshooting."""

    def __init__(self):
        self.events: List[MemorySnapshot] = []

    def record(self, label: str, echo: bool = True, logger=print) -> MemorySnapshot:
        snap = take_snapshot(label)
        self.events.append(snap)
        if echo:
            logger("  " + snap.line())
        return snap

    def peak_total_pss(self) -> Optional[float]:
        vals = [e.total_pss for e in self.events if e.total_pss is not None]
        return max(vals) / (1024 * 1024) if vals else None

    def peak_cgroup(self) -> Optional[float]:
        vals = [e.cgroup_current for e in self.events if e.cgroup_current is not None]
        return max(vals) / (1024 * 1024) if vals else None

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.events], f, indent=2)
