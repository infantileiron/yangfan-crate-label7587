"""Instrument registry, wash-log validation, and chronological batch holdout."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Sensor:
    key: str
    code: str
    label: str
    station: str
    lower: float
    upper: float
    pilot: str


# Registry order breaks exact mutual-information ties in favor of direct readings.
SENSORS = (
    Sensor("spray_pressure_bar", "PT-21", "喷淋压力（bar）", "主洗喷淋总管", 0, 20,
           "在主洗总管保留压力变送器，按单筐记录喷淋阶段平均压力"),
    Sensor("pump_current_a", "CT-21", "喷淋泵电流（A）", "主洗泵控制柜", 0, 80,
           "复用控制柜电流采样，将主洗泵稳态电流写入单筐日志"),
    Sensor("rinse_temp_c", "TT-32", "末漂温度（°C）", "末漂喷嘴前端", 0, 110,
           "在末漂喷嘴前保留温度探头，记录每筐末漂阶段平均温度"),
    Sensor("dryer_dewpoint_c", "DT-41", "烘干露点（°C）", "烘干出口风道", -60, 80,
           "在出口风道试装露点探头，将出筐时的读数绑定筐次"),
    Sensor("label_dwell_min", "SC-11", "贴标静置（min）", "贴标与进洗扫码站", 0, 1440,
           "复用两站扫码时间戳，自动记录贴标至进洗的静置分钟数"),
    Sensor("conductivity_ms_cm", "EC-31", "末漂电导率（mS/cm）", "末漂回流管", 0, 30,
           "在末漂回流管试装电导率探头，按筐次采集补水稳定后的读数"),
)


@dataclass(frozen=True)
class SelectionOptions:
    holdout_start: date = date(2026, 11, 23)
    bins: int = 4
    redundancy_threshold: float = 0.90
    min_mi: float = 0.02
    max_sensors: int = 3


@dataclass(frozen=True)
class WashRecord:
    cycle_id: str
    batch_id: str
    wash_date: date
    detached: int
    readings: tuple[float, ...]


@dataclass(frozen=True)
class HoldoutSplit:
    train: list[WashRecord]
    holdout: list[WashRecord]
    excluded: list[WashRecord]


def _iso_date(value: object, label: str) -> date:
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value.strip())
        except ValueError:
            pass
        else:
            if parsed.isoformat() == value.strip():
                return parsed
    raise ValueError(f"{label}须使用 YYYY-MM-DD 格式，请填写有效日期。")


def _number(value: object, label: str, lower: float, upper: float) -> float:
    if type(value) not in (int, float) or not lower <= value <= upper:
        raise ValueError(f"{label}须为 {lower:g} 到 {upper:g} 之间的有限数值，请核对原始读数或配置。")
    return float(value)


def _integer(value: object, label: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label}须为 {lower} 到 {upper} 之间的整数，请修正后重试。")
    return value


def parse_options(options: dict) -> SelectionOptions:
    if not isinstance(options, dict):
        raise ValueError("分析配置须为键值对象，请重新填写筛选参数。")
    defaults = SelectionOptions()
    return SelectionOptions(
        holdout_start=_iso_date(options.get("holdout_start", defaults.holdout_start.isoformat()), "留出起始日"),
        bins=_integer(options.get("bins", defaults.bins), "分箱数", 2, 8),
        redundancy_threshold=_number(options.get("redundancy_threshold", defaults.redundancy_threshold),
                                     "相关阈值", 0.50, 1.0),
        min_mi=_number(options.get("min_mi", defaults.min_mi), "最低互信息", 0, 1),
        max_sensors=_integer(options.get("max_sensors", defaults.max_sensors), "最多保留测点", 1, len(SENSORS)),
    )


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空，请补填洗筐日志中的原始编号。")
    return value.strip()


def validate_rows(rows: list[dict]) -> list[WashRecord]:
    if not isinstance(rows, list) or len(rows) < 12:
        raise ValueError("至少需要 12 条洗筐记录，请补充至少 8 条训练记录和 4 条留出记录。")
    records = []
    seen_cycles = set()
    for line, row in enumerate(rows, start=2):
        if not isinstance(row, dict):
            raise ValueError(f"CSV 第 {line} 行缺少字段，请按演示模板重新导入。")
        cycle_id = _identifier(row.get("cycle_id"), f"CSV 第 {line} 行的洗筐记录号")
        if cycle_id in seen_cycles:
            raise ValueError(f"洗筐记录号 {cycle_id} 重复，请核对扫码日志并删除重复行。")
        seen_cycles.add(cycle_id)
        records.append(WashRecord(
            cycle_id=cycle_id,
            batch_id=_identifier(row.get("batch_id"), f"记录 {cycle_id} 的洗筐批次"),
            wash_date=_iso_date(row.get("wash_date"), f"记录 {cycle_id} 的洗筐日期"),
            detached=_integer(row.get("detached"), f"记录 {cycle_id} 的标签脱落标记", 0, 1),
            readings=tuple(_number(row.get(sensor.key), f"记录 {cycle_id} 的{sensor.label}",
                                   sensor.lower, sensor.upper) for sensor in SENSORS),
        ))
    return sorted(records, key=lambda row: (row.wash_date, row.cycle_id))


def lock_holdout(records: list[WashRecord], start: date) -> HoldoutSplit:
    # If a batch spans the cutoff, its earlier rows are isolated from both sets.
    holdout = [row for row in records if row.wash_date >= start]
    holdout_batches = {row.batch_id for row in holdout}
    train, excluded = [], []
    for row in records:
        if row.wash_date < start:
            partition = excluded if row.batch_id in holdout_batches else train
            partition.append(row)

    for label, partition, minimum in (("训练集", train, 8), ("留出集", holdout, 4)):
        if len(partition) < minimum or len({row.batch_id for row in partition}) < 2:
            raise ValueError(f"{label}至少需要 {minimum} 条记录和 2 个独立批次。"
                             "请补充完整批次，或在查看筛选结果前重新锁定有效起始日。")
        if len({row.detached for row in partition}) < 2:
            raise ValueError(f"{label}须同时含有留存和脱落记录，请补齐该日期范围内的质检结果。")
    return HoldoutSplit(train, holdout, excluded)
