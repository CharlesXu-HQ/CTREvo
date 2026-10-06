"""Independent scoring; paired intervals do not correct adaptive model selection."""

import numpy as np
from sklearn.metrics import roc_auc_score


def checked(labels, prediction):
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    if (labels.ndim != 1 or prediction.shape != labels.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(prediction).all()
            or np.any((prediction < 0) | (prediction > 1))):
        raise ValueError('need aligned binary labels and finite probability vectors in [0,1]')
    return labels.astype(np.float64), np.clip(prediction.astype(np.float64), 1e-7, 1 - 1e-7)


def losses(labels, prediction):
    y, p = checked(labels, prediction)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def evaluate(labels, prediction):
    y, p = checked(labels, prediction)
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = []
    for index in range(10):
        mask = bins == index
        if mask.any():
            calibration.append({'bin': index, 'rows': int(mask.sum()),
                                'predicted': float(p[mask].mean()), 'observed': float(y[mask].mean())})
    return {'rows': len(y), 'logloss': float(losses(y, p).mean()),
            'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'brier': float(np.square(y - p).mean()), 'observed_ctr': float(y.mean()),
            'mean_prediction': float(p.mean()), 'calibration': calibration}


def paired_logloss(labels, baseline, candidate):
    difference = losses(labels, candidate) - losses(labels, baseline)
    mean = float(difference.mean())
    radius = 1.96 * float(difference.std(ddof=1)) / np.sqrt(len(difference)) if len(difference) > 1 else 0
    return {'mean': mean, 'lower': mean - radius, 'upper': mean + radius, 'confidence': .95,
            'direction': 'candidate minus baseline; negative favors candidate',
            'method': 'paired per-impression normal interval',
            'limitations': 'exploratory; no adaptive-selection correction; anonymous rows prevent user clustering'}
