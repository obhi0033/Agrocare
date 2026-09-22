import cv2
import numpy as np


def _safe_hist(channel, bins, rng):
    h = cv2.calcHist([channel], [0], None, [bins], rng).flatten().astype(np.float32)
    s = float(h.sum())
    return h / s if s else h


def extract_features_legacy(img):
    """Original 11-feature extractor kept so the existing model still works."""
    im = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    green = ((h >= 35) & (h <= 90) & (s > 40)).mean()
    yellowbrown = ((h >= 8) & (h <= 35) & (s > 40)).mean()
    dark = (v < 80).mean()
    edges = cv2.Canny(gray, 80, 160)
    return np.array([h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std(),
                     green, yellowbrown, dark, (edges > 0).mean(), gray.std()], dtype=np.float32)


def extract_features(img):
    """Improved handcrafted features: HSV/Lab histograms, color ratios, texture and edges."""
    im = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    L, a, b = cv2.split(lab)

    feats = []
    # Global channel statistics.
    for ch in (h, s, v, L, a, b, gray):
        feats += [float(ch.mean()), float(ch.std()),
                  float(np.percentile(ch, 25)), float(np.percentile(ch, 50)), float(np.percentile(ch, 75))]

    # Normalized histograms capture disease-related color distribution better than means alone.
    feats.extend(_safe_hist(h, 18, [0, 180]))
    feats.extend(_safe_hist(s, 16, [0, 256]))
    feats.extend(_safe_hist(v, 16, [0, 256]))
    feats.extend(_safe_hist(a, 16, [0, 256]))
    feats.extend(_safe_hist(b, 16, [0, 256]))

    # Useful plant/lesion color proportions.
    masks = [
        (h >= 35) & (h <= 90) & (s > 40),      # green
        (h >= 18) & (h < 35) & (s > 40),       # yellow
        (h >= 5) & (h < 18) & (s > 45),        # brown/orange
        (v < 70),                               # dark lesion-like pixels
        (s < 35) & (v > 150),                   # pale/whitish
        (s > 90) & (v < 140),                   # strongly colored dark tissue
    ]
    feats.extend([float(m.mean()) for m in masks])

    # Texture / edge features at two scales.
    for k in (3, 7):
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        residual = cv2.absdiff(gray, blur)
        feats += [float(residual.mean()), float(residual.std())]
    for lo, hi in ((40, 100), (80, 160)):
        edges = cv2.Canny(gray, lo, hi)
        feats.append(float((edges > 0).mean()))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    feats += [float(gray.std()), float(lap.var())]

    # Spatial color summaries (2x2 grid) retain coarse lesion location/distribution.
    for yy in range(2):
        for xx in range(2):
            patch = hsv[yy*128:(yy+1)*128, xx*128:(xx+1)*128]
            ph, ps, pv = cv2.split(patch)
            feats += [float(ph.mean()), float(ps.mean()), float(pv.mean()), float(pv.std())]

    return np.asarray(feats, dtype=np.float32)


def features_for_model(img, model):
    """Use legacy features for the old 11-feature model; improved features after retraining."""
    n = getattr(model, "n_features_in_", None)
    if n is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            n = getattr(step, "n_features_in_", None)
            if n is not None:
                break
    if n == 11:
        return extract_features_legacy(img)
    return extract_features(img)
