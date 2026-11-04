"""
Light-weight segmentation code (OpenCV + NumPy + PIL only).

Copied verbatim from beans_pipeline.py so the deployed app segments images exactly
like the training pipeline did, without needing PyTorch.
"""
import cv2
import numpy as np
from PIL import Image

SIZE = 224
LESION_MIN_SCORE = 12.0  # same value used when the models were trained

# central ellipse that segment() falls back to when GrabCut fails (leaf mask unreliable)
FALLBACK_MASK = np.zeros((SIZE, SIZE), np.uint8)
cv2.ellipse(FALLBACK_MASK, (SIZE // 2, SIZE // 2), (int(.35 * SIZE), int(.40 * SIZE)), 0, 0, 360, 1, -1)


def _centre_component(binary, h, w):
    """Keep the connected component that overlaps the image centre the most."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return binary
    centre = lab[int(.3 * h):int(.7 * h), int(.3 * w):int(.7 * w)]
    counts = np.bincount(centre[centre > 0].ravel(), minlength=n)
    best = int(counts.argmax()) if counts.sum() > 0 else int(1 + stats[1:, cv2.CC_STAT_AREA].argmax())
    return (lab == best).astype(np.uint8)


def segment(rgb: np.ndarray):
    """Return (leaf_mask, lesion_mask) as uint8 {0,1} arrays.

    Leaf  : GrabCut seeded with a central rectangle (the photographed leaf is
            roughly centred), keep only the central connected component.
    Lesion: pixels inside the (eroded) leaf whose Lab colour deviates strongly
            from THAT leaf's own median colour (robust MAD threshold).
    """
    h, w = rgb.shape[:2]

    # ---- leaf
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gc = np.zeros((h, w), np.uint8)
    rect = (int(.10 * w), int(.10 * h), int(.80 * w), int(.80 * h))
    try:
        cv2.grabCut(bgr, gc, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
        leaf = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        leaf = np.zeros((h, w), np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    leaf = _centre_component(leaf, h, w)
    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf = np.zeros((h, w), np.uint8)
    cv2.drawContours(leaf, cnts, -1, 1, thickness=cv2.FILLED)  # fill holes (lesions inside)
    if leaf.mean() < 0.10:  # GrabCut failed -> fall back to a central ellipse
        leaf[:] = 0
        cv2.ellipse(leaf, (w // 2, h // 2), (int(.35 * w), int(.40 * h)), 0, 0, 360, 1, -1)

    # ---- lesions: colour deviation from this leaf's own median colour
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    inner = cv2.erode(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # drop edge pixels
    if inner.sum() < 100:
        inner = leaf
    med = np.median(lab[inner == 1], axis=0)
    d = lab - med
    score = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2 + 0.5 * d[..., 0] ** 2)
    s_in = score[inner == 1]
    mad = np.median(np.abs(s_in - np.median(s_in))) + 1e-6
    thr = max(np.median(s_in) + 3 * 1.4826 * mad, LESION_MIN_SCORE)
    lesion = ((score > thr) & (inner == 1)).astype(np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lb, stats, _ = cv2.connectedComponentsWithStats(lesion)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 15 or a > 0.25 * leaf.sum():  # too small = noise, too big = shadow/lighting
            lesion[lb == i] = 0
    return leaf, lesion


def make_variant(pil_img: Image.Image, mode: str) -> Image.Image:
    rgb = np.array(pil_img.convert("RGB").resize((SIZE, SIZE)))
    if mode == "raw":
        return Image.fromarray(rgb)
    leaf, lesion = segment(rgb)
    if mode == "leaf":  # background removed, whole leaf kept
        out = rgb * leaf[..., None]
    elif mode == "lesion":  # lesions + small context ring; rest blacked out
        ctx = cv2.dilate(lesion, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        out = rgb * ctx[..., None]
    else:
        raise ValueError(mode)
    return Image.fromarray(out.astype(np.uint8))
