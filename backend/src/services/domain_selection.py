"""Training-only quantile binning, mutual information, pruning, and validation."""

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from math import fsum, log, log2, sqrt

from .domain_validation import SENSORS, SelectionOptions, Sensor, WashRecord


@dataclass(frozen=True)
class FeatureScore:
    sensor: Sensor
    index: int
    cuts: tuple[float, ...]
    mi: float


@dataclass(frozen=True)
class Screening:
    feature: FeatureScore
    status: str
    substitute: Sensor | None = None
    correlation: float | None = None


@dataclass(frozen=True)
class Validation:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @property
    def recall(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def balanced_accuracy(self) -> float:
        specificity = self.true_negative / (self.true_negative + self.false_positive)
        return (self.recall + specificity) / 2


def quantile_cuts(values: tuple[float, ...], bins: int) -> tuple[float, ...]:
    """Use the nearest distinct-value boundary to each requested rank, ties lower."""
    ordered = sorted(values)
    boundaries = [index for index in range(1, len(ordered)) if ordered[index - 1] < ordered[index]]
    if not boundaries:
        return ()
    cuts = set()
    for step in range(1, bins):
        target = len(ordered) * step / bins
        index = bisect_left(boundaries, target)
        neighbors = boundaries[max(0, index - 1):index + 1]
        boundary = min(neighbors, key=lambda rank: (abs(rank - target), rank))
        cuts.add(ordered[boundary - 1])
    return tuple(sorted(cuts))


def mutual_information(codes: tuple[int, ...], labels: tuple[int, ...]) -> float:
    """Empirical I(X;Y) in bits, using occupied cells of the contingency table."""
    size = len(labels)
    feature_counts, label_counts = Counter(codes), Counter(labels)
    joint = Counter(zip(codes, labels))
    return max(0.0, fsum(
        count / size * log2(count * size / (feature_counts[code] * label_counts[label]))
        for (code, label), count in joint.items()
    ))


def absolute_pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_mean, right_mean = fsum(left) / len(left), fsum(right) / len(right)
    x = [value - left_mean for value in left]
    y = [value - right_mean for value in right]
    x_scale, y_scale = max(map(abs, x)), max(map(abs, y))
    if x_scale == 0 or y_scale == 0:
        return 0.0
    x = [value / x_scale for value in x]
    y = [value / y_scale for value in y]
    denominator = sqrt(fsum(value ** 2 for value in x) * fsum(value ** 2 for value in y))
    return min(1.0, abs(fsum(a * b for a, b in zip(x, y))) / denominator)


def screen_features(train: list[WashRecord], options: SelectionOptions) -> list[Screening]:
    columns = list(zip(*(row.readings for row in train)))
    labels = tuple(row.detached for row in train)
    scores = []
    for index, (sensor, values) in enumerate(zip(SENSORS, columns)):
        cuts = quantile_cuts(values, options.bins)
        codes = tuple(bisect_left(cuts, value) for value in values)
        scores.append(FeatureScore(sensor, index, cuts, mutual_information(codes, labels)))
    scores.sort(key=lambda score: -score.mi)

    selected: list[FeatureScore] = []
    screenings = []
    for score in scores:
        if not score.cuts:
            screenings.append(Screening(score, "恒定读数"))
            continue
        if score.mi == 0 or score.mi < options.min_mi:
            screenings.append(Screening(score, "信息不足"))
            continue

        correlations = [(absolute_pearson(columns[score.index], columns[prior.index]), prior)
                        for prior in selected]
        strongest = max(correlations, key=lambda item: item[0], default=None)
        if strongest and strongest[0] >= options.redundancy_threshold:
            correlation, prior = strongest
            screenings.append(Screening(score, "相关冗余", prior.sensor, correlation))
        elif len(selected) == options.max_sensors:
            screenings.append(Screening(score, "超出测点预算"))
        else:
            selected.append(score)
            screenings.append(Screening(score, "入选"))
    return screenings


def _confusion(labels: list[int], predictions: list[int]) -> Validation:
    counts = Counter(zip(labels, predictions))
    return Validation(counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])


def validate_selection(train: list[WashRecord], holdout: list[WashRecord],
                       features: list[FeatureScore]) -> tuple[Validation, Validation]:
    """Fit Laplace-smoothed categorical naive Bayes, then score locked rows."""
    counts = Counter(row.detached for row in train)
    log_priors = tuple(log((counts[label] + 1) / (len(train) + 2)) for label in (0, 1))
    likelihoods = []
    for feature in features:
        observed = Counter((row.detached, bisect_left(feature.cuts, row.readings[feature.index]))
                           for row in train)
        bin_count = len(feature.cuts) + 1
        likelihoods.append(tuple(
            tuple(log((observed[label, code] + 1) / (counts[label] + bin_count))
                  for code in range(bin_count)) for label in (0, 1)
        ))

    predictions = []
    for row in holdout:
        scores = list(log_priors)
        for feature, likelihood in zip(features, likelihoods):
            code = bisect_left(feature.cuts, row.readings[feature.index])
            for label in (0, 1):
                scores[label] += likelihood[label][code]
        predictions.append(int(scores[1] >= scores[0]))
    labels = [row.detached for row in holdout]
    majority = int(counts[1] >= counts[0])
    return _confusion(labels, predictions), _confusion(labels, [majority] * len(holdout))
