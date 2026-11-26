"""Command-line entry point for the cross-shift rolling pilot analysis.

Examples
--------
# Run the built-in demo log with the locked parameters.
python -m scripts.rolling_pilot

# Resume after an interruption; checkpoints make the rerun incremental.
python -m scripts.rolling_pilot --state-dir /tmp/rolling-state

# Point at an exported wash-log CSV instead of the simulated data.
python -m scripts.rolling_pilot --csv chengwan_crate_wash_20261117-24.csv

# Fault-injection: crash after N verified windows, then rerun to verify resume.
python -m scripts.rolling_pilot --fail-after 1
python -m scripts.rolling_pilot
"""

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.domain import sample_rows  # noqa: E402
from services.domain_rolling_store import RollingStore  # noqa: E402

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "src" / "services" / "domain.json"


def _convert(value, field):
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        raise ValueError("不能为空")
    kind = field["type"]
    if kind == "str":
        return str(value).strip()
    number = int(str(value)) if kind == "int" else float(value)
    if not math.isfinite(number):
        raise ValueError("必须为有限数值")
    if number < field.get("min", -math.inf) or number > field.get("max", math.inf):
        raise ValueError("数值超出可接受范围")
    return number


def _load_csv(path: Path) -> list[dict]:
    """Parse the wash-log CSV with the stdlib only (no database dependencies)."""
    profile = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    columns = profile["columns"]
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"), newline=""), strict=True)
    required = {field["key"] for field in columns}
    if not reader.fieldnames or set(reader.fieldnames) != required:
        raise ValueError("CSV 表头须为 " + "、".join(field["key"] for field in columns))
    rows = []
    for raw in reader:
        if None in raw:
            raise ValueError(f"第 {reader.line_num} 行列数多于表头。")
        normalized = {}
        for field in columns:
            try:
                normalized[field["key"]] = _convert(raw[field["key"]], field)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"第 {reader.line_num} 行 {field['label']} 无效（{error}）。") from error
        rows.append(normalized)
    if not rows:
        raise ValueError("CSV 需要至少一条数据。")
    return rows


def _options(args) -> dict:
    options = {}
    if args.holdout_start:
        options["holdout_start"] = args.holdout_start
    for key in ("bins", "redundancy_threshold", "min_mi", "max_sensors"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    return options


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="按批次完结日的滚动窗口测点试点分析")
    parser.add_argument("--csv", type=Path, help="洗筐 CSV；缺省时使用内置演示数据")
    parser.add_argument("--state-dir", type=Path, default=Path("artifacts/rolling-state"),
                        help="断点检查点与完整结果目录")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"),
                        help="汇总 JSON/CSV 导出目录")
    parser.add_argument("--holdout-start", help="留出首日 YYYY-MM-DD")
    parser.add_argument("--bins", type=int)
    parser.add_argument("--redundancy-threshold", dest="redundancy_threshold", type=float)
    parser.add_argument("--min-mi", dest="min_mi", type=float)
    parser.add_argument("--max-sensors", dest="max_sensors", type=int)
    parser.add_argument("--fail-after", type=int, default=0,
                        help="核验完前 N 个窗口后中断（仅用于验证断点续跑）")
    args = parser.parse_args(argv)

    rows = _load_csv(args.csv) if args.csv else sample_rows()
    options = _options(args)
    store = RollingStore(args.state_dir)

    interrupted = {"raise": args.fail_after}

    def progress(done, total, resumed):
        tag = "续跑命中" if resumed else "已核验"
        print(f"[{done}/{total}] {tag}窗口", file=sys.stderr)
        if not resumed and interrupted["raise"] and done >= interrupted["raise"]:
            raise RuntimeError(f"模拟中断：已在第 {done} 个窗口后停止，用于验证续跑。")

    try:
        report = store.run(rows, options, progress=progress)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = report["fingerprint"]
    json_path = args.artifact_dir / f"rolling-pilot-{fingerprint[:12]}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    from utils.exports import write_csv
    tables = {
        "windows": report["windows"],
        "selected-per-window": report["selected_per_window"],
        "sensor-rows": report["sensor_rows"],
        "combo-audit": report["combo_audit"],
    }
    csv_paths = []
    for name, table in tables.items():
        if not table:
            continue
        target = args.artifact_dir / f"rolling-pilot-{fingerprint[:12]}-{name}.csv"
        target.write_bytes(write_csv(table, list(table[0])))
        csv_paths.append(str(target))

    print(json.dumps({
        "headline": report["headline"],
        "can_publish": report["can_publish"],
        "locked": report["locked"],
        "lock_reason": report["lock_reason"],
        "selected_codes": report["selected_codes"],
        "windows": report["scope"]["窗口数"],
        "candidate_combos": report["candidate_count"],
        "report_json": str(json_path),
        "report_csv": csv_paths,
    }, ensure_ascii=False, indent=2))
    return 0 if report["can_publish"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
