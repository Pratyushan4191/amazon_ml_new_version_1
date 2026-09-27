"""
Shared worker-pool management for the BER pipeline.

Why this module exists
----------------------
Before this, every call site did its own `mp.Pool(mp.cpu_count() - 1)` and
tore it down again at the end of the call. In Stage 4 that meant one pool
construction per feature chunk -- at production scale that is one pool per
2,000 raw S1 rows, i.e. >1,000 pool create/teardown cycles per split.

Two problems with that, in order of severity:

1. It fought the memory design. A forked child inherits the parent's address
   space copy-on-write. CPython touches (and therefore dirties) the header page
   of essentially every live object as refcounts change, so forking N workers
   from a parent holding a multi-GB sources table converts a large fraction of
   that table from shared-shared into shared-private, N times over. Forking the
   pool *before* the big tables are loaded keeps the children cheap, and
   reusing one pool means that cost is paid once per run instead of once per
   chunk.

2. Fixed overhead per call. Pool construction, plus the pickling of every task
   argument, is charged against small chunks that would be faster to run
   inline. `compute_serial_threshold` below routes small work around the pool
   entirely.

Worker count
------------
Defaults to *all* logical CPUs (`os.cpu_count()`), because the parent process
is blocked on `pool.imap` for essentially the whole duration of both the
blocking and feature stages -- it is not competing for a core. Reserve a core
with `BER_WORKERS` if you need the parent responsive (e.g. to run a memory
sampler thread, or on a box where the parent does meaningful work between
`imap` calls).

`BER_WORKERS` overrides the default. `BER_SPAWN=1` switches the start method
to `spawn`, which shares nothing with the parent at all -- the right choice
when the parent heap is unavoidably large and COW accounting is a concern, at
the cost of a slower one-time startup and re-importing the module in each
worker.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from typing import Any, Callable, Iterable, List, Optional

# Below this many rows, fork+pickle overhead exceeds the cost of the work
# itself, so callers should just run serially in-process. Tuned against the
# measured Stage 4 profile: the per-row fuzzy-match work is ~90us/row/core, so
# a few thousand rows is already a few hundred ms of real work -- well above
# the ~1-3ms it costs to ship a task to a worker.
SERIAL_THRESHOLD = 4_000

_POOL: Optional["mp.pool.Pool"] = None
_POOL_WORKERS: Optional[int] = None
_POOL_METHOD: Optional[str] = None
_POOL_INIT: Optional[Callable[..., None]] = None


def resolve_workers(requested: Optional[int] = None) -> int:
    """
    Number of worker processes to use.

    Precedence: explicit argument > ``BER_WORKERS`` env var > bounded default.
    On large cloud instances (e.g. 64-96 vCPUs), spawning one worker per vCPU
    causes memory exhaustion from COW degradation and per-worker runtime state.
    Therefore, the automatic default is capped at 16 workers unless explicitly
    overridden. Always at least 1.
    """
    if requested is not None:
        return max(1, int(requested))
    env = os.environ.get("BER_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpus = os.cpu_count() or 1
    # Bounded default for production stability on high-core AWS instances (e.g. 64/96 vCPUs)
    return max(1, min(cpus, 16))


def should_run_serial(n_items: int, workers: int) -> bool:
    """True when `n_items` is small enough that a pool is not worth it."""
    if workers <= 1:
        return True
    return n_items < SERIAL_THRESHOLD


def _start_method() -> str:
    if "fork" not in mp.get_all_start_methods():
        return "spawn"
    return "spawn" if os.environ.get("BER_SPAWN") == "1" else "fork"


def get_pool(
    initializer: Optional[Callable[..., None]] = None,
    initargs: tuple = (),
    workers: Optional[int] = None,
):
    """
    Return the process-wide pool, creating it on first use.

    The pool is a singleton: repeated calls with the same worker count and
    start method return the same pool, so per-call fork cost is paid once per
    process. Passing a *different* `initializer` than the pool was built with
    raises, because silently ignoring it would produce workers holding stale
    state -- a class of bug that is very hard to see from the output.
    """
    global _POOL, _POOL_WORKERS, _POOL_METHOD, _POOL_INIT

    method = _start_method()
    if method == "fork" and "fork" not in mp.get_all_start_methods():
        method = mp.get_start_method()

    want_workers = resolve_workers(workers)

    if _POOL is not None:
        if (_POOL_WORKERS, _POOL_METHOD, initializer) != (
                want_workers, method, _POOL_INIT):
            _shutdown_pool()
        else:
            return _POOL

    ctx = mp.get_context(method)
    _POOL = ctx.Pool(
        processes=want_workers,
        initializer=initializer,
        initargs=initargs,
    )
    _POOL_WORKERS = want_workers
    _POOL_METHOD = method
    _POOL_INIT = initializer
    return _POOL


def shutdown_pool():
    """Tear down the singleton pool, if one exists. Safe to call repeatedly."""
    _shutdown_pool()


def _shutdown_pool():
    global _POOL, _POOL_WORKERS, _POOL_METHOD, _POOL_INIT
    if _POOL is not None:
        try:
            _POOL.terminate()
            _POOL.join()
        except Exception:
            pass
    _POOL = None
    _POOL_WORKERS = None
    _POOL_METHOD = None
    _POOL_INIT = None


atexit.register(_shutdown_pool)


def imap_tasks(
    func: Callable,
    tasks: Iterable[Any],
    workers: Optional[int] = None,
    initializer: Optional[Callable[..., None]] = None,
    initargs: tuple = (),
    serial_threshold: int = SERIAL_THRESHOLD,
    chunksize: int = 1,
):
    """
    Map `func` over `tasks`, using the shared pool when it is worth it and
    running inline when it is not.

    Returns a list of results in task order either way, so callers never have
    to care which path was taken.
    """
    task_list: List[Any] = list(tasks)
    n = len(task_list)
    n_workers = resolve_workers(workers)

    if n_workers <= 1 or n < serial_threshold:
        return [func(t) for t in task_list]

    pool = get_pool(
        initializer=initializer, initargs=initargs, workers=n_workers
    )
    # imap (not imap_unordered) so result order matches task order; the caller
    # relies on positional alignment between rows and feature blocks.
    return list(pool.imap(func, task_list, chunksize=chunksize))
