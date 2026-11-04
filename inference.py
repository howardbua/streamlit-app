"""
Inference without PyTorch: ONNX Runtime + the segmentation code from leafseg.py.

Model files (raw.onnx, leaf.onnx, classes.json) are looked up in this order:
  1. a local  models/  folder next to this file
  2. the Hugging Face model repo named in HF_REPO (downloaded once, then cached)
"""
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

import leafseg

# EDIT THIS: your Hugging Face MODEL repo that holds raw.onnx, leaf.onnx and classes.json
HF_REPO = os.environ.get("HF_REPO", "Howard/bean_leaf_onnx")

MODES = ("raw", "leaf")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _resolve(filename):
    local = os.path.join("models", filename)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=filename)


def load_models():
    """Create the ONNX sessions once (kept lean on memory) and read the class names."""
    import json
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False   # do not hold on to big scratch buffers
    so.enable_mem_pattern = False
    so.intra_op_num_threads = 2
    sessions = {
        m: ort.InferenceSession(_resolve(f"{m}.onnx"), so, providers=["CPUExecutionProvider"])
        for m in MODES
    }
    with open(_resolve("classes.json")) as f:
        classes = json.load(f)
    return sessions, classes


def _prep(pil_variant):
    """Same as torchvision ToTensor + Normalize used in training."""
    a = np.asarray(pil_variant.convert("RGB"), np.float32) / 255.0
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def analyze(sessions, classes, img):
    img = ImageOps.exif_transpose(img).convert("RGB")  # fix phone-photo rotation
    img.thumbnail((1024, 1024))                         # bound memory/time for big uploads

    probs = {}
    for mode, sess in sessions.items():
        x = _prep(leafseg.make_variant(img, mode))
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        probs[mode] = _softmax(logits.astype(np.float64))

    rgb = np.array(img.resize((leafseg.SIZE, leafseg.SIZE)))
    leaf, lesion = leafseg.segment(rgb)
    fallback = bool(np.array_equal(leaf, leafseg.FALLBACK_MASK))
    severity = 100.0 * float(lesion.sum()) / max(float(leaf.sum()), 1.0)

    outline = rgb.copy()
    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline, cnts, -1, (255, 255, 0), 2)

    raw_label = classes[int(probs["raw"].argmax())]
    leaf_label = classes[int(probs["leaf"].argmax())]
    return {
        "probs": probs,
        "raw_label": raw_label,
        "leaf_label": leaf_label,
        "agree": raw_label == leaf_label,
        "outline": outline,
        "leaf_input": (rgb * leaf[..., None]).astype(np.uint8),
        "fallback": fallback,
        "severity": severity,
    }
