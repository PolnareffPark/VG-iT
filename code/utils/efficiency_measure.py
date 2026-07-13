"""Efficiency profiling utilities shared across VG-iT and baseline entry points.

Provides EfficiencyCollector for capturing per-iter latency, VRAM peaks
(reserved + allocated), and batch=1 inference latency under a single,
reproducible protocol. Used by run.py / experiments / baselines so that
all 13 models record identical metric definitions.

Reference: plans/streamed-inventing-dove.md (approved 2026-04-14).
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np
import torch


def _is_cuda(device: torch.device) -> bool:
    return getattr(device, "type", None) == "cuda"


def pre_allocate_dummy_tensors_train(
    device: torch.device,
    batch_size: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    N: int,
    has_marks: bool,
    seed: int = 2021,
) -> dict:
    """Pre-allocate GPU dummy tensors ONCE for reuse across dummy training iters.

    Purpose: eliminate CPU-side ``torch.randn`` + H2D copy + allocator churn
    from the dummy measurement bracket. Enables true "pure GPU iteration"
    semantics per efficiency framework v6 spec.

    Contract:
      - All tensors created directly on ``device`` with ``dtype=torch.float32``.
      - Deterministic per seed for reproducibility across repetitions
        (CPU generator → ``.to(device)``).
      - ``has_marks=False`` → ``x_mark``, ``y_mark`` are ``None`` (matches real
        path's PEMS/Solar branch that sets ``batch_x_mark = None`` before
        model forward).
      - ``dec_inp`` is precomputed (mirrors ``torch.zeros_like`` +
        ``torch.cat`` pattern that real path does per-iter; dummy does once
        since ``y`` is reused).
      - Returned tensors MUST NOT be mutated in-place by the training loop.

    Returns:
        Dict with keys:
          - ``'x'``:       shape ``(batch_size, seq_len, N)``
          - ``'y'``:       shape ``(batch_size, label_len + pred_len, N)``
          - ``'x_mark'``:  shape ``(batch_size, seq_len, 4)`` or ``None``
          - ``'y_mark'``:  shape ``(batch_size, label_len + pred_len, 4)`` or ``None``
          - ``'dec_inp'``: shape ``(batch_size, label_len + pred_len, N)``
                           (zeros for last ``pred_len`` slice, copy of
                           ``y[:, :label_len, :]`` for prefix)
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    total_len = int(label_len) + int(pred_len)
    dtype = torch.float32

    x_cpu = torch.randn(
        int(batch_size), int(seq_len), int(N), generator=gen, dtype=dtype
    )
    y_cpu = torch.randn(
        int(batch_size), total_len, int(N), generator=gen, dtype=dtype
    )
    x = x_cpu.to(device=device, dtype=dtype)
    y = y_cpu.to(device=device, dtype=dtype)

    if has_marks:
        x_mark_cpu = torch.randn(
            int(batch_size), int(seq_len), 4, generator=gen, dtype=dtype
        )
        y_mark_cpu = torch.randn(
            int(batch_size), total_len, 4, generator=gen, dtype=dtype
        )
        x_mark = x_mark_cpu.to(device=device, dtype=dtype)
        y_mark = y_mark_cpu.to(device=device, dtype=dtype)
    else:
        x_mark = None
        y_mark = None

    # dec_inp: prefix = y[:, :label_len, :], suffix = zeros of length pred_len
    # mirrors real path: torch.zeros_like(y[:, -pred_len:, :]) +
    # torch.cat([y[:, :label_len, :], zeros], dim=1)
    zeros_tail = torch.zeros(
        int(batch_size), int(pred_len), int(N), device=device, dtype=dtype
    )
    dec_inp = torch.cat([y[:, : int(label_len), :], zeros_tail], dim=1)

    return {
        "x": x,
        "y": y,
        "x_mark": x_mark,
        "y_mark": y_mark,
        "dec_inp": dec_inp,
    }


def pre_allocate_dummy_tensors_test(
    device: torch.device,
    batch_size: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    N: int,
    has_marks: bool,
    seed: int = 2021,
) -> dict:
    """Identical shape and contract to :func:`pre_allocate_dummy_tensors_train`.

    Separate symbol for clarity and future divergence (e.g. if test ever uses
    a different ``batch_size``). Currently delegates to the train variant.
    """
    return pre_allocate_dummy_tensors_train(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        N=N,
        has_marks=has_marks,
        seed=seed,
    )


def pre_allocate_dummy_tensors_crossformer(
    device: torch.device,
    batch_size: int,
    in_len: int,
    out_len: int,
    N: int,
    seed: int = 2021,
) -> dict:
    """Crossformer variant: 2-tuple (batch_x, batch_y) only.

    Crossformer uses ``args.in_len`` / ``args.out_len`` / ``args.data_dim``
    (not ``seq_len`` / ``pred_len`` / ``N``) and does not consume time-marks
    or a decoder-input tensor.

    Returns:
        Dict with keys:
          - ``'x'``: shape ``(batch_size, in_len, N)``
          - ``'y'``: shape ``(batch_size, out_len, N)``
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    dtype = torch.float32

    x_cpu = torch.randn(
        int(batch_size), int(in_len), int(N), generator=gen, dtype=dtype
    )
    y_cpu = torch.randn(
        int(batch_size), int(out_len), int(N), generator=gen, dtype=dtype
    )
    x = x_cpu.to(device=device, dtype=dtype)
    y = y_cpu.to(device=device, dtype=dtype)

    return {"x": x, "y": y}


class EfficiencyCollector:
    """Collect train/infer latency and VRAM peak under a single protocol.

    Time stats:
      - Real data: epoch 0 iters excluded from latency stats (warmup).
      - Dummy data: first ``dummy_warmup_iters`` excluded.
      - VRAM peak captured over the full run (warmup included).

    All times stored in milliseconds. VRAM in GB.
    """

    def __init__(
        self,
        device: torch.device,
        mode: str = "real",
        dummy_warmup_iters: int = 20,
    ) -> None:
        if mode not in ("real", "dummy"):
            raise ValueError(f"mode must be 'real' or 'dummy', got {mode!r}")
        self.device = device
        self.mode = mode
        self.dummy_warmup_iters = int(dummy_warmup_iters)

        self.train_iter_times_ms: list[float] = []
        self.train_iter_epochs: list[int] = []
        self.infer_iter_times_ms: list[float] = []

        self.train_vram_reserved_GB: float = 0.0
        self.train_vram_allocated_GB: float = 0.0
        self.infer_vram_reserved_GB: float = 0.0
        self.infer_vram_allocated_GB: float = 0.0

        self._train_t0: Optional[float] = None
        self._infer_t0: Optional[float] = None

        self.protocol_version: str = (
            "v6_pure_compute_dummy" if mode == "dummy"
            else "v6_full_pipeline_real"
        )

    # -------- VRAM control --------
    def reset_train_vram(self) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def reset_infer_vram(self) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def capture_train_vram_peak(self) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)  # pending op 완료 보장
            self.train_vram_reserved_GB = (
                torch.cuda.max_memory_reserved(self.device) / 1024**3
            )
            self.train_vram_allocated_GB = (
                torch.cuda.max_memory_allocated(self.device) / 1024**3
            )

    def capture_infer_vram_peak(self) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)  # pending op 완료 보장
            self.infer_vram_reserved_GB = (
                torch.cuda.max_memory_reserved(self.device) / 1024**3
            )
            self.infer_vram_allocated_GB = (
                torch.cuda.max_memory_allocated(self.device) / 1024**3
            )

    # -------- Per-iter timing --------
    def begin_train_iter(self) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)
        self._train_t0 = time.perf_counter()

    def end_train_iter(self, epoch_idx: int) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)
        if self._train_t0 is None:
            return
        elapsed_ms = (time.perf_counter() - self._train_t0) * 1000.0
        self.train_iter_times_ms.append(elapsed_ms)
        self.train_iter_epochs.append(int(epoch_idx))
        self._train_t0 = None

    def begin_infer_iter(self) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)
        self._infer_t0 = time.perf_counter()

    def end_infer_iter(self) -> None:
        if _is_cuda(self.device):
            torch.cuda.synchronize(self.device)
        if self._infer_t0 is None:
            return
        elapsed_ms = (time.perf_counter() - self._infer_t0) * 1000.0
        self.infer_iter_times_ms.append(elapsed_ms)
        self._infer_t0 = None

    # -------- Stats reduction --------
    def _train_times_for_stats(self) -> list[float]:
        if self.mode == "dummy":
            if len(self.train_iter_times_ms) > self.dummy_warmup_iters:
                return self.train_iter_times_ms[self.dummy_warmup_iters :]
            return self.train_iter_times_ms[:]
        # real: drop epoch 0
        return [
            t
            for t, e in zip(self.train_iter_times_ms, self.train_iter_epochs)
            if e > 0
        ]

    def _infer_times_for_stats(self) -> list[float]:
        if self.mode == "dummy":
            if len(self.infer_iter_times_ms) > self.dummy_warmup_iters:
                return self.infer_iter_times_ms[self.dummy_warmup_iters :]
            return self.infer_iter_times_ms[:]
        return self.infer_iter_times_ms[:]

    @staticmethod
    def _summarize(arr: list[float], prefix: str) -> dict:
        # Always return the same three keys so JSON records merge consistently
        # across pandas; None marks missing measurements instead of dropping the
        # columns entirely.
        if not arr:
            return {
                f"{prefix}_median": None,
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
            }
        a = np.asarray(arr, dtype=np.float64)
        return {
            f"{prefix}_median": round(float(np.median(a)), 6),
            f"{prefix}_mean": round(float(np.mean(a)), 6),
            f"{prefix}_std": round(float(np.std(a, ddof=0)), 6),
        }

    def to_dict(self) -> dict:
        train_times = self._train_times_for_stats()
        infer_times = self._infer_times_for_stats()

        result: dict = {
            "data_source": self.mode,
            "train_vram_reserved_GB_peak": round(self.train_vram_reserved_GB, 6),
            "train_vram_allocated_GB_peak": round(self.train_vram_allocated_GB, 6),
            "infer_vram_reserved_GB_peak": round(self.infer_vram_reserved_GB, 6),
            "infer_vram_allocated_GB_peak": round(self.infer_vram_allocated_GB, 6),
            "total_train_iters_recorded": len(self.train_iter_times_ms),
            "train_iters_used_for_stats": len(train_times),
            "total_infer_iters_recorded": len(self.infer_iter_times_ms),
            "infer_iters_used_for_stats": len(infer_times),
        }

        # Real-mode iters_per_epoch: count epoch 0 iters
        if self.mode == "real" and self.train_iter_epochs:
            iters_e0 = sum(1 for e in self.train_iter_epochs if e == 0)
            result["iters_per_epoch_real"] = iters_e0
        else:
            result["iters_per_epoch_real"] = None

        result.update(self._summarize(train_times, "train_latency_ms"))
        result.update(self._summarize(infer_times, "infer_latency_ms"))

        result["protocol_version"] = self.protocol_version

        return result


def collect_environment_metadata(device: torch.device, seed: int = 2021) -> dict:
    """Return reproducibility metadata to embed at the top of each output JSON."""
    meta: dict = {
        "seed": int(seed),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    try:
        meta["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        meta["cudnn_version"] = None
    if _is_cuda(device):
        meta["gpu_name"] = torch.cuda.get_device_name(device)
        meta["gpu_memory_GB"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**3, 2
        )
    else:
        meta["gpu_name"] = "cpu"
        meta["gpu_memory_GB"] = 0
    return meta


def atomic_append_json_record(path: str, record: dict, metadata: Optional[dict] = None) -> None:
    """Append a record into a JSON file atomically.

    Schema: ``{"metadata": {...}, "records": [...]}``. If ``path`` does not
    exist, it is created with the supplied ``metadata`` (required on first call).
    Subsequent calls only append to ``records``; ``metadata`` is preserved.

    Atomicity: writes to a temp file beside the target then ``os.replace``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            payload = {"metadata": metadata or {}, "records": []}
    else:
        if metadata is None:
            raise ValueError(
                "metadata required when creating a new efficiency JSON file"
            )
        payload = {"metadata": metadata, "records": []}

    payload.setdefault("records", []).append(record)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)
