"""Durable, resumable checkpoints for the rolling pilot analysis.

The rolling analysis can span many windows and may be interrupted (worker
restart, deployment, failure).  Each *analysis* is identified by a fingerprint
of its normalized rows plus the locked parameters.  Window results are flushed
to a checkpoint file as soon as they are verified, so a resumed run continues
from the first window not yet on disk instead of recomputing finished work.

A POSIX ``fcntl`` lock serializes every run for the same analysis: concurrent
resumes take turns against one shared checkpoint and only the runner that
completes the final window produces the complete published result.  Callers
that simply want the answer always receive that single complete report.
"""

import fcntl
import json
import os
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .domain_rolling import (
    WindowResult,
    build_windows,
    evaluate_window,
    select_pilot,
)
from .domain_validation import SelectionOptions, parse_options, validate_rows


def analysis_fingerprint(rows: list[dict], options: dict) -> str:
    payload = json.dumps(
        {"rows": rows, "options": options},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _stats_to_dict(stats) -> list[dict]:
    return [
        {
            "code": stat.code,
            "index": stat.index,
            "cuts": list(stat.cuts),
            "bin_count": stat.bin_count,
            "mi": stat.mi,
            "status": stat.status,
            "substitute": stat.substitute,
            "correlation": stat.correlation,
        }
        for stat in stats
    ]


def _combo_to_dict(combo) -> dict:
    return {
        "codes": list(combo.codes),
        "admissible": combo.admissible,
        "rejected_reason": combo.rejected_reason,
        "model": asdict(combo.model) if combo.model else None,
        "baseline": asdict(combo.baseline) if combo.baseline else None,
        "balanced_accuracy": combo.balanced_accuracy,
        "baseline_accuracy": combo.baseline_accuracy,
        "improvement": combo.improvement,
    }


def _window_to_dict(result: WindowResult) -> dict:
    window = result.window
    return {
        "window": {
            "sequence": window.sequence,
            "completion_day": window.completion_day.isoformat(),
            "holdout_batch": window.holdout_batch,
            # Training/holdout rows are reconstructed from the source data on
            # resume; only their identity is persisted for cross-checking.
            "train_cycles": [row.cycle_id for row in window.train],
            "holdout_cycles": [row.cycle_id for row in window.holdout],
        },
        "summary": {
            "train_rows": result.train_rows,
            "train_batches": result.train_batches,
            "holdout_rows": result.holdout_rows,
            "train_positive": result.train_positive,
            "holdout_positive": result.holdout_positive,
            "holdout_negative": result.holdout_negative,
            "class_complete": result.class_complete,
            "blocked_reason": result.blocked_reason,
        },
        "sensor_stats": _stats_to_dict(result.sensor_stats),
        "combos": [_combo_to_dict(combo) for combo in result.combos],
    }


class RollingStore:
    """File-backed checkpoint directory with an exclusive per-analysis lock."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _paths(self, fingerprint: str) -> tuple[Path, Path, Path]:
        base = self.directory / f"rolling-{fingerprint[:16]}"
        return base.with_suffix(".lock"), base.with_suffix(".json"), base.with_suffix(".report.json")

    def _read_checkpoint(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _atomic_write(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, allow_nan=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            temporary = tmp.name
        os.replace(temporary, path)

    def run(self, rows: list[dict], options: dict, progress=None) -> dict:
        """Execute or resume one analysis, returning the single complete report.

        ``progress(done, total, resumed)`` is optional and invoked after each
        window is durably checkpointed.
        """
        fingerprint = analysis_fingerprint(rows, options)
        lock_path, checkpoint_path, report_path = self._paths(fingerprint)
        config = parse_options(options)
        records = validate_rows(rows)
        windows = build_windows(records, config.holdout_start)

        with open(lock_path, "w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                finished = report_path.exists()
                checkpoint = self._read_checkpoint(checkpoint_path)
                done_windows: dict[int, WindowResult] = {}
                if checkpoint:
                    self._guard_fingerprint(checkpoint, fingerprint, config, records)
                    done_windows = self._load_windows(checkpoint, windows)
                if finished and len(done_windows) == len(windows):
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if progress:
                        progress(len(windows), len(windows), True)
                    return report

                for window in windows:
                    if window.sequence in done_windows:
                        continue
                    result = evaluate_window(window, config)
                    done_windows[window.sequence] = result
                    self._atomic_write(checkpoint_path, self._build_checkpoint(
                        fingerprint, config, records, done_windows, windows, complete=False))
                    if progress:
                        progress(len(done_windows), len(windows), False)
                    # A hard gate failure (missing class) stops later windows;
                    # the locked decision still records every verified window.
                    if result.blocked_reason:
                        break

                ordered = [done_windows[window.sequence] for window in windows
                           if window.sequence in done_windows]
                decision = select_pilot(ordered, config)
                from .domain_rolling import assemble_report
                report = assemble_report(records, config, ordered, decision)
                report["fingerprint"] = fingerprint
                self._atomic_write(checkpoint_path, self._build_checkpoint(
                    fingerprint, config, records,
                    {item.window.sequence: item for item in ordered}, windows, complete=True))
                self._atomic_write(report_path, report)
                return report
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _build_checkpoint(self, fingerprint, config: SelectionOptions, records,
                          done_windows: dict[int, WindowResult], windows, complete: bool) -> dict:
        return {
            "version": 1,
            "fingerprint": fingerprint,
            "complete": complete,
            "parameters": {
                "holdout_start": config.holdout_start.isoformat(),
                "bins": config.bins,
                "redundancy_threshold": config.redundancy_threshold,
                "min_mi": config.min_mi,
                "max_sensors": config.max_sensors,
            },
            "data_checksum": sha256(
                json.dumps([row.cycle_id for row in records], ensure_ascii=False).encode()
            ).hexdigest(),
            "window_count": len(windows),
            "windows": [_window_to_dict(done_windows[window.sequence])
                        for window in windows if window.sequence in done_windows],
        }

    def _guard_fingerprint(self, checkpoint: dict, fingerprint: str,
                           config: SelectionOptions, records) -> None:
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("检查点与当前数据或参数不一致，不能混入另一次分析，请更换分析标识后重跑。")
        expected = {
            "holdout_start": config.holdout_start.isoformat(),
            "bins": config.bins,
            "redundancy_threshold": config.redundancy_threshold,
            "min_mi": config.min_mi,
            "max_sensors": config.max_sensors,
        }
        if checkpoint.get("parameters") != expected:
            raise ValueError("检查点记录的锁定参数与本次运行不一致。")

    def _load_windows(self, checkpoint: dict, windows) -> dict[int, WindowResult]:
        from .domain_selection import Validation
        from .domain_rolling import ComboResult, SensorStat

        window_by_seq = {window.sequence: window for window in windows}
        loaded: dict[int, WindowResult] = {}
        for payload in checkpoint["windows"]:
            seq = payload["window"]["sequence"]
            if seq not in window_by_seq:
                raise ValueError(f"检查点中的窗口 {seq} 在当前数据中不存在，无法续跑。")
            window = window_by_seq[seq]
            if [r.cycle_id for r in window.holdout] != payload["window"]["holdout_cycles"]:
                raise ValueError(f"窗口 {seq} 的留出筐次与检查点不一致。")
            summary = payload["summary"]
            stats = [
                SensorStat(
                    code=item["code"], index=item["index"], cuts=tuple(item["cuts"]),
                    bin_count=item["bin_count"], mi=item["mi"], status=item["status"],
                    substitute=item["substitute"], correlation=item["correlation"],
                )
                for item in payload["sensor_stats"]
            ]
            combos = []
            for item in payload["combos"]:
                model = Validation(**item["model"]) if item["model"] else None
                baseline = Validation(**item["baseline"]) if item["baseline"] else None
                combos.append(ComboResult(
                    codes=tuple(item["codes"]), admissible=item["admissible"],
                    rejected_reason=item["rejected_reason"], model=model, baseline=baseline,
                    balanced_accuracy=item["balanced_accuracy"],
                    baseline_accuracy=item["baseline_accuracy"],
                    improvement=item["improvement"],
                ))
            loaded[seq] = WindowResult(
                window=window, sensor_stats=stats, combos=combos, **summary
            )
        return loaded
