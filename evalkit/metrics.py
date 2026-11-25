"""Metric maths. Plain Python so every number can be explained and checked by hand."""
import math
from collections import Counter


def classification_report(y_true, y_pred, labels):
    """Accuracy, per-class precision/recall/F1, macro F1, confusion matrix."""
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    per, f1s = {}, []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = tp + fn
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    confusion = {t: Counter() for t in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
    return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s), "per_class": per,
            "confusion": {t: dict(c) for t, c in confusion.items()}, "n": n}


def field_prf(pairs):
    """Field-level precision / recall / F1.

    `pairs` is a list of (expected_fields, actual_fields).
    TP: expected value present and equal.  FN: expected value present but not returned correctly.
    FP: a value was returned that is wrong, or was returned where nothing was expected.
    """
    tp = fp = fn = 0
    for exp, act in pairs:
        for f, e in exp.items():
            a = act.get(f)
            if e is not None and a == e:
                tp += 1
            elif e is not None:
                fn += 1
                if a is not None:
                    fp += 1
            elif a is not None:
                fp += 1
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": tp + fn}


def wilson_interval(k, n, z=1.96):
    """95% confidence interval for a proportion (better than +/- sqrt(p(1-p)/n) for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_p(p1, n1, p2, n2):
    """Two-sided p-value: are two proportions really different, or is it sampling noise?"""
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0 if p1 == p2 else 0.0
    z = abs(p1 - p2) / se
    return math.erfc(z / math.sqrt(2))


def cohens_kappa(a, b):
    """Agreement between two labelers beyond what chance would give. 0 = chance, 1 = perfect."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
