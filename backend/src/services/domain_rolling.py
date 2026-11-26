"""Walk-forward batch-holdout pilot screening with one locked sensor set.

The quality group pilots the *same* measuring points across shifts.  Instead of a
single holdout cut, every batch that finishes on or after the locked first
holdout day becomes its own expanding window:

* a window's holdout is one whole batch (a batch is never split);
* its training set uses only batches whose completion day is strictly earlier;
* mutual-information and correlation gates, the quantile binning and the
  Laplace-smoothed naive Bayes are reused unchanged from the single-window
  method and are refit on each window's training data.

Every sensor combination within the budget is evaluated on every window.  The
published combination is the globally admissible one whose *weakest* window
improves balanced accuracy over the training majority-class baseline the most;
exact ties prefer fewer sensors, then the earlier registry order.  A window
missing either label class, or a chosen combination that never beats the
baseline on some window, locks publication.
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations

from .domain_selection import (
    FeatureScore,
    Validation,
    absolute_pearson,
    mutual_information,
    quantile_cuts,
    screen_features,
    validate_selection,
)
from .domain_validation import (
    SENSORS,
    SelectionOptions,
    WashRecord,
    parse_options,
    validate_rows,
)

# Registry index order is reused to break ties deterministically.
SENSOR_ORDER = {sensor.code: index for index, sensor in enumerate(SENSORS)}


@dataclass(frozen=True)
class Window:
    """One expanding walk-forward window anchored on a single holdout batch."""

    sequence: int
    completion_day: date
    holdout_batch: str
    train: list[WashRecord]
    holdout: list[WashRecord]


@dataclass(frozen=True)
class SensorStat:
    """Per-window training-only diagnostics for one candidate sensor."""

    code: str
    index: int
    cuts: tuple[float, ...]
    bin_count: int
    mi: float
    status: str
    substitute: str
    correlation: float | None


@dataclass(frozen=True)
class ComboResult:
    """How one sensor combination behaves on one window's locked holdout."""

    codes: tuple[str, ...]
    admissible: bool
    rejected_reason: str
    model: Validation | None
    baseline: Validation | None
    balanced_accuracy: float | None
    baseline_accuracy: float | None
    improvement: float | None


@dataclass
class WindowResult:
    window: Window
    train_rows: int
    train_batches: int
    holdout_rows: int
    train_positive: int
    holdout_positive: int
    holdout_negative: int
    class_complete: bool
    blocked_reason: str
    sensor_stats: list[SensorStat] = field(default_factory=list)
    combos: list[ComboResult] = field(default_factory=list)


def build_windows(records: list[WashRecord], holdout_start: date) -> list[Window]:
    """Group whole batches by completion day and lay out expanding windows.

    A batch's completion day is the latest wash date it contains, so a batch
    spanning the cutoff stays together and only enters training after the whole
    batch has finished.  Training for every window is strictly older than the
    holdout batch's completion day.
    """
    completion: dict[str, date] = {}
    members: dict[str, list[WashRecord]] = {}
    for row in records:
        completion[row.batch_id] = max(completion.get(row.batch_id, row.wash_date), row.wash_date)
        members.setdefault(row.batch_id, []).append(row)

    ordered_batches = sorted(members, key=lambda batch: (completion[batch], batch))
    first_index = next(
        (index for index, batch in enumerate(ordered_batches) if completion[batch] >= holdout_start),
        None,
    )
    if first_index is None:
        raise ValueError("留出首日及之后没有完整批次，请补采后续班次或重新锁定留出首日。")

    windows: list[Window] = []
    for sequence, holdout_index in enumerate(range(first_index, len(ordered_batches)), start=1):
        holdout_batch = ordered_batches[holdout_index]
        day = completion[holdout_batch]
        train = [
            row
            for index in range(holdout_index)
            for row in members[ordered_batches[index]]
            if completion[ordered_batches[index]] < day
        ]
        holdout = list(members[holdout_batch])
        windows.append(Window(sequence, day, holdout_batch, train, holdout))
    return windows


def _feature_for(train: list[WashRecord], sensor_index: int, bins: int) -> FeatureScore:
    values = tuple(row.readings[sensor_index] for row in train)
    labels = tuple(row.detached for row in train)
    cuts = quantile_cuts(values, bins)
    codes = tuple(bisect_left(cuts, value) for value in values)
    return FeatureScore(SENSORS[sensor_index], sensor_index, cuts, mutual_information(codes, labels))


def _sensor_statistics(train: list[WashRecord], options: SelectionOptions) -> list[SensorStat]:
    """Reuse the single-window screening to derive the MI and redundancy gates."""
    stats = [
        SensorStat(
            code=item.feature.sensor.code,
            index=item.feature.index,
            cuts=item.feature.cuts,
            bin_count=len(item.feature.cuts) + 1,
            mi=item.feature.mi,
            status=item.status,
            substitute=item.substitute.code if item.substitute else "",
            correlation=item.correlation,
        )
        for item in screen_features(train, options)
    ]
    stats.sort(key=lambda stat: SENSOR_ORDER[stat.code])
    return stats


def _pairwise_redundant(train: list[WashRecord], indices: tuple[int, ...],
                        threshold: float) -> tuple[int, int, float] | None:
    columns = list(zip(*(row.readings for row in train)))
    worst = None
    for left, right in combinations(indices, 2):
        correlation = absolute_pearson(columns[left], columns[right])
        if correlation >= threshold and (worst is None or correlation > worst[2]):
            worst = (left, right, correlation)
    return worst


def evaluate_window(window: Window, options: SelectionOptions) -> WindowResult:
    """Run every in-budget combination through one window."""
    train, holdout = window.train, window.holdout
    train_positive = sum(row.detached for row in train)
    holdout_positive = sum(row.detached for row in holdout)
    result = WindowResult(
        window=window,
        train_rows=len(train),
        train_batches=len({row.batch_id for row in train}),
        holdout_rows=len(holdout),
        train_positive=train_positive,
        holdout_positive=holdout_positive,
        holdout_negative=len(holdout) - holdout_positive,
        class_complete=(0 < train_positive < len(train)) and (0 < holdout_positive < len(holdout)),
        blocked_reason="",
    )
    if not result.class_complete:
        missing = []
        if train_positive == 0 or train_positive == len(train):
            missing.append("训练集")
        if holdout_positive == 0 or holdout_positive == len(holdout):
            missing.append("留出批次")
        result.blocked_reason = "、".join(missing) + "缺少留存或脱落实例"
        return result

    stats = _sensor_statistics(train, options)
    result.sensor_stats = stats
    columns = list(zip(*(row.readings for row in train)))
    eligible = [stat.index for stat in stats if stat.status != "恒定读数"
                and stat.mi >= options.min_mi]
    single_reason = {
        stat.index: ("训练集恒定读数" if stat.status == "恒定读数"
                     else f"互信息 {stat.mi:.4f} bit 低于门槛 {options.min_mi:g}")
        for stat in stats
    }

    for size in range(1, options.max_sensors + 1):
        for indices in combinations(sorted(eligible), size):
            redundant = _pairwise_redundant(train, indices, options.redundancy_threshold)
            if redundant is not None:
                left, right, correlation = redundant
                result.combos.append(ComboResult(
                    codes=tuple(SENSORS[i].code for i in indices),
                    admissible=False,
                    rejected_reason=(f"{SENSORS[left].code} 与 {SENSORS[right].code} 相关 "
                                     f"{correlation:.4f} 达到去冗余门槛"),
                    model=None, baseline=None, balanced_accuracy=None,
                    baseline_accuracy=None, improvement=None,
                ))
                continue
            features = [_feature_for(train, index, options.bins) for index in indices]
            model, baseline = validate_selection(train, holdout, features)
            improvement = model.balanced_accuracy - baseline.balanced_accuracy
            result.combos.append(ComboResult(
                codes=tuple(SENSORS[i].code for i in indices),
                admissible=True,
                rejected_reason="",
                model=model,
                baseline=baseline,
                balanced_accuracy=model.balanced_accuracy,
                baseline_accuracy=baseline.balanced_accuracy,
                improvement=improvement,
            ))
    # Record gate-rejected singletons so the audit explains every sensor.
    for stat in stats:
        if stat.index not in eligible:
            result.combos.append(ComboResult(
                codes=(stat.code,),
                admissible=False,
                rejected_reason=single_reason[stat.index],
                model=None, baseline=None, balanced_accuracy=None,
                baseline_accuracy=None, improvement=None,
            ))
    result.combos.sort(key=lambda combo: (len(combo.codes), tuple(SENSOR_ORDER[c] for c in combo.codes)))
    return result


def _combo_key(codes: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(SENSOR_ORDER[code] for code in codes)


def select_pilot(results: list[WindowResult], options: SelectionOptions) -> dict:
    """Pick the globally admissible combination with the strongest weak window."""
    blocked = [item for item in results if item.blocked_reason]
    if blocked:
        first = blocked[0]
        return {
            "locked": True,
            "lock_reason": f"窗口 {first.window.sequence}（{first.window.holdout_batch}）{first.blocked_reason}",
            "selected_codes": [],
            "windows": _summary(results),
        }

    by_codes: dict[tuple[str, ...], list[ComboResult]] = {}
    for item in results:
        for combo in item.combos:
            by_codes.setdefault(combo.codes, []).append(combo)

    candidates = []
    for codes, per_window in by_codes.items():
        if len(per_window) != len(results) or not all(combo.admissible for combo in per_window):
            continue
        improvements = [combo.improvement for combo in per_window]
        weakest_index = min(range(len(improvements)), key=lambda i: (improvements[i], i))
        candidates.append({
            "codes": codes,
            "size": len(codes),
            "key": _combo_key(codes),
            "min_improvement": improvements[weakest_index],
            "weakest_window": weakest_index + 1,
            "improvements": improvements,
            "balanced_accuracy": [combo.balanced_accuracy for combo in per_window],
            "baseline_accuracy": [combo.baseline_accuracy for combo in per_window],
        })

    if not candidates:
        return {
            "locked": True,
            "lock_reason": "没有任何测点组合能在全部窗口同时通过互信息与相关门槛。",
            "selected_codes": [],
            "windows": _summary(results),
        }

    candidates.sort(key=lambda item: (-item["min_improvement"], item["size"], item["key"]))
    best = candidates[0]
    beats_everywhere = all(value > 0 for value in best["improvements"])
    selected = best["codes"]
    return {
        "locked": not beats_everywhere,
        "lock_reason": "" if beats_everywhere else
        f"最弱窗口（窗口 {best['weakest_window']}）平衡准确率与基线持平或落后，未取得正向提升。",
        "selected_codes": list(selected),
        "selection_rule": ("在预算内、且每个窗口训练集都通过互信息与相关门槛的组合中，"
                           "取最弱窗口平衡准确率相对多数类基线提升最大者；持平先取测点更少者，"
                           "再按测点登记顺序。"),
        "candidate_count": len(candidates),
        "ranked": [
            {
                "codes": list(item["codes"]),
                "测点数量": item["size"],
                "最弱窗口": item["weakest_window"],
                "最弱窗口提升": item["min_improvement"],
                "各窗提升": item["improvements"],
            }
            for item in candidates
        ],
        "windows": _summary(results),
        "best": best,
    }


def _summary(results: list[WindowResult]) -> list[dict]:
    rows = []
    for item in results:
        window = item.window
        rows.append({
            "窗口": window.sequence,
            "批次完结日": window.completion_day.isoformat(),
            "留出批次": window.holdout_batch,
            "训练筐次": item.train_rows,
            "训练批次": item.train_batches,
            "留出筐次": item.holdout_rows,
            "留出脱落": item.holdout_positive,
            "留出留存": item.holdout_negative,
            "类别齐备": item.class_complete,
            "阻断原因": item.blocked_reason,
        })
    return rows


def run_rolling(rows: list[dict], options: dict) -> dict:
    """Pure entry point used by both the CLI runner and the service layer."""
    config = parse_options(options)
    records = validate_rows(rows)
    windows = build_windows(records, config.holdout_start)
    results = [evaluate_window(window, config) for window in windows]
    decision = select_pilot(results, config)
    return assemble_report(records, config, results, decision)


def assemble_report(records: list[WashRecord], config: SelectionOptions,
                    results: list[WindowResult], decision: dict) -> dict:
    completion_days = sorted({_completion(records, batch) for batch in {row.batch_id for row in records}})
    window_rows = decision["windows"]
    selected_codes = tuple(decision.get("selected_codes", []))

    combo_audit = []
    for item in results:
        for combo in item.combos:
            combo_audit.append({
                "窗口": item.window.sequence,
                "留出批次": item.window.holdout_batch,
                "测点组合": "、".join(combo.codes),
                "测点数": len(combo.codes),
                "是否通过门槛": "是" if combo.admissible else "否",
                "门槛说明": combo.rejected_reason,
                "平衡准确率（%）": round(combo.balanced_accuracy * 100, 2)
                if combo.balanced_accuracy is not None else "",
                "基线平衡准确率（%）": round(combo.baseline_accuracy * 100, 2)
                if combo.baseline_accuracy is not None else "",
                "相对基线提升（百分点）": round(combo.improvement * 100, 2)
                if combo.improvement is not None else "",
            })

    sensor_rows = []
    for item in results:
        for stat in item.sensor_stats:
            sensor_rows.append({
                "窗口": item.window.sequence,
                "留出批次": item.window.holdout_batch,
                "测点编号": stat.code,
                "互信息（bit）": round(stat.mi, 6),
                "有效分箱": stat.bin_count,
                "门槛状态": stat.status,
                "关联测点": stat.substitute,
            })

    selected_per_window = []
    best = decision.get("best")
    if best:
        for index, item in enumerate(results):
            combo = next((c for c in item.combos if c.codes == tuple(best["codes"])), None)
            if combo and combo.model:
                selected_per_window.append({
                    "窗口": item.window.sequence,
                    "留出批次": item.window.holdout_batch,
                    "测点组合": "、".join(combo.codes),
                    "平衡准确率（%）": round(combo.balanced_accuracy * 100, 2),
                    "基线平衡准确率（%）": round(combo.baseline_accuracy * 100, 2),
                    "相对基线提升（百分点）": round(combo.improvement * 100, 2),
                    "正确识别留存": combo.model.true_negative,
                    "留存误报脱落": combo.model.false_positive,
                    "脱落漏报留存": combo.model.false_negative,
                    "正确识别脱落": combo.model.true_positive,
                })

    can_publish = not decision["locked"] and bool(selected_codes)
    if can_publish:
        headline = (f"跨 {len(results)} 个批次窗口锁定同一组合 {'、'.join(selected_codes)}，"
                    f"最弱窗口提升 {best['min_improvement'] * 100:.2f} 个百分点，建议发布试点")
    else:
        headline = "滚动窗口未全部通过发布门槛，锁住发布并保留全部窗口核验结果"

    return {
        "headline": headline,
        "can_publish": can_publish,
        "locked": decision["locked"],
        "lock_reason": decision["lock_reason"],
        "analysis": "rolling-batch-holdout",
        "parameters": {
            "holdout_start": config.holdout_start.isoformat(),
            "bins": config.bins,
            "redundancy_threshold": config.redundancy_threshold,
            "min_mi": config.min_mi,
            "max_sensors": config.max_sensors,
        },
        "scope": {
            "总筐次": len(records),
            "总批次": len({row.batch_id for row in records}),
            "批次完结日": [day.isoformat() for day in completion_days],
            "窗口数": len(results),
            "候选测点数": len(SENSORS),
        },
        "selected_codes": list(selected_codes),
        "selection_rule": decision.get("selection_rule", ""),
        "candidate_count": decision.get("candidate_count", 0),
        "windows": window_rows,
        "selected_per_window": selected_per_window,
        "ranked_candidates": decision.get("ranked", []),
        "sensor_rows": sensor_rows,
        "combo_audit": combo_audit,
        "notes": [
            "锁定筐次与参数后，从留出首日起按批次完结日逐窗留出整批；每窗训练集只含更早完结的批次，"
            "跨日批次按最晚洗筐日归为完结日，绝不拆批。",
            f"各窗沿用互信息门槛 {config.min_mi:g} bit、绝对 Pearson 相关门槛 "
            f"{config.redundancy_threshold:g}、{config.bins} 个目标分位箱与 {config.max_sensors} 路预算，"
            "分箱、互信息与朴素贝叶斯均只在该窗训练集上拟合。",
            "枚举预算内全部测点组合，仅保留在每个窗口训练集都通过互信息与相关门槛的全局组合；"
            "按最弱窗口相对多数类基线的平衡准确率提升排序，持平先取测点更少、再按登记顺序。",
            "任一窗口缺少留存或脱落实例，或选中组合在某窗未严格胜过多数类基线，即锁住发布。",
            "窗口结果逐窗落盘并绑定数据与参数指纹；中断后续跑从已核验窗口继续，同一分析并发续跑只产出一份完整结果。",
        ],
    }


def _completion(records: list[WashRecord], batch_id: str) -> date:
    return max(row.wash_date for row in records if row.batch_id == batch_id)
