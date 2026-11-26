"""Run the crate-wash sensor screening and issue a QA pilot disposition."""

from hashlib import sha256
import json

from .domain_seed import SEED, sample_rows as sample_rows
from .domain_selection import Screening, screen_features, validate_selection
from .domain_validation import HoldoutSplit, lock_holdout, parse_options, validate_rows


def _lock_fingerprint(split: HoldoutSplit) -> str:
    membership = {
        name: [(row.wash_date.isoformat(), row.batch_id, row.cycle_id) for row in group]
        for name, group in (("train", split.train), ("holdout", split.holdout), ("excluded", split.excluded))
    }
    payload = json.dumps(membership, ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _recommendation(screening: Screening, can_publish: bool) -> str:
    sensor = screening.feature.sensor
    if screening.status == "入选":
        if can_publish:
            return f"纳入质控试点。{sensor.pilot}。"
        return f"挂起 {sensor.code} 采购，补采后续独立批次并复核留出成绩。"
    if screening.substitute:
        if can_publish:
            return f"暂缓单独立项，以 {screening.substitute.code} 作为本次质量筛选的试点测点。"
        return f"记录与 {screening.substitute.code} 的训练集冗余，复测后再决定测点取舍。"
    if screening.status == "恒定读数":
        return f"先检查 {sensor.station} 的采集链路和量程，补齐变化工况后再筛。"
    if screening.status == "超出测点预算":
        return f"列入追加候补，释放测点名额后再复核 {sensor.code}。"
    return f"暂缓 {sensor.code} 试点，保留现有日志供后续批次复核。"


def compute(rows: list[dict], options: dict) -> dict:
    config = parse_options(options)
    split = lock_holdout(validate_rows(rows), config.holdout_start)
    screenings = screen_features(split.train, config)
    selected = [item.feature for item in screenings if item.status == "入选"]
    model, baseline = validate_selection(split.train, split.holdout, selected)
    can_publish = bool(selected) and model.balanced_accuracy > baseline.balanced_accuracy
    scores = []
    for item in screenings:
        feature = item.feature
        scores.append({
            "测点编号": feature.sensor.code,
            "候选变量": feature.sensor.label,
            "安装位置": feature.sensor.station,
            "互信息（bit）": round(feature.mi, 6),
            "有效分箱": len(feature.cuts) + 1,
            "筛选状态": item.status,
            "关联测点": item.substitute.code if item.substitute else "",
            "绝对相关系数": round(item.correlation, 6) if item.correlation is not None else "",
            "质控建议": _recommendation(item, can_publish),
        })
    selected_codes = "、".join(feature.sensor.code for feature in selected)
    if can_publish:
        headline = f"{selected_codes} 通过批次留出验证，建议纳入质控试点"
    elif selected:
        headline = f"{selected_codes} 尚未超过留出基线，测点采购暂缓"
    else:
        headline = "当前阈值下无测点入选，待补测或复核筛选配置"

    return {
        "headline": headline,
        "metrics": {
            "训练筐次": len(split.train),
            "训练批次": len({row.batch_id for row in split.train}),
            "留出筐次": len(split.holdout),
            "留出批次": len({row.batch_id for row in split.holdout}),
            "隔离筐次": len(split.excluded),
            "候选测点": len(scores),
            "入选测点": len(selected),
            "相关冗余测点": sum(item.status == "相关冗余" for item in screenings),
            "留出平衡准确率（%）": round(model.balanced_accuracy * 100, 2),
            "多数类基线平衡准确率（%）": round(baseline.balanced_accuracy * 100, 2),
            "留出脱落召回率（%）": round(model.recall * 100, 2),
            "正确识别留存": model.true_negative,
            "留存误报脱落": model.false_positive,
            "脱落漏报留存": model.false_negative,
            "正确识别脱落": model.true_positive,
        },
        "rows": scores,
        "notes": [
            f"留出自 {config.holdout_start.isoformat()} 起（含当日）。训练日期为 "
            f"{split.train[0].wash_date} 至 {split.train[-1].wash_date}，留出日期为 "
            f"{split.holdout[0].wash_date} 至 {split.holdout[-1].wash_date}。"
            "跨界批次只把起始日及之后的记录计入留出，较早记录隔离。",
            f"分组锁定 SHA-256 {_lock_fingerprint(split)}。选项调参请使用训练数据，留出应只用于最终复核。",
            f"训练集按 {config.bins} 个目标分位箱离散化，再按互信息（bit）降序排列。"
            "并列时优先登记在前的直接测点。互信息反映关联，采购前仍须现场试装。",
            f"最低互信息 {config.min_mi:g} bit，绝对 Pearson 相关达到 {config.redundancy_threshold:g} 即去冗余，"
            f"最多保留 {config.max_sensors} 个测点。",
            "用训练分箱和拉普拉斯平滑朴素贝叶斯预测脱落。平衡准确率为脱落和留存召回率的平均值。"
            "入选方案在留出集上高于训练多数类基线才可发布，概率并列时判为脱落。",
            f"演示日志使用固定模拟种子 {SEED}，现场试点需用实际洗筐日志重新验证。",
        ],
        "chart": {
            "label": "候选测点与标签脱落的训练集互信息",
            "unit": "bit",
            "points": [{"label": row["测点编号"], "value": row["互信息（bit）"]} for row in scores],
        },
        "can_publish": can_publish,
    }
