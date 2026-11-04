import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import inference

st.set_page_config(page_title="Bean leaf disease: before vs after segmentation", layout="wide")


@st.cache_resource(show_spinner="Loading models (the first start can take a minute)...")
def get_models():
    return inference.load_models()


st.title("Bean leaf disease: classification before vs after segmentation")
st.write(
    "Upload a photo of a bean leaf. The app classifies the original image and the "
    "segmented image (background removed), and shows both results side by side."
)

file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
if file is None:
    st.info("Upload a photo to begin. A close-up of a single leaf works best.")
    st.stop()

try:
    img = Image.open(file)
except Exception:
    st.error("That file could not be read as an image. Please try a JPG or PNG photo.")
    st.stop()

sessions, classes = get_models()
with st.spinner("Analysing..."):
    res = inference.analyze(sessions, classes, img)

if res["agree"]:
    st.success(f"Before and after segmentation agree: **{res['raw_label']}**")
else:
    st.warning(
        f"Before and after segmentation disagree: raw says **{res['raw_label']}**, "
        f"leaf-segmented says **{res['leaf_label']}**."
    )
if res["fallback"]:
    st.warning(
        "Automatic leaf segmentation failed on this image, so the segmented-image result "
        "is unreliable. Trust the raw result more."
    )

col1, col2 = st.columns(2)
for col, mode, title in [
    (col1, "raw", "Before segmentation (raw image)"),
    (col2, "leaf", "After segmentation (leaf only)"),
]:
    with col:
        st.subheader(title)
        p = res["probs"][mode]
        k = int(np.argmax(p))
        st.markdown(f"**{classes[k]}**  ({p[k] * 100:.1f}% confidence)")
        st.bar_chart(pd.DataFrame({"probability": p}, index=classes))

st.subheader("What the models saw")
c1, c2, c3 = st.columns(3)
c1.image(np.array(img.convert("RGB")), caption="Your photo")
c2.image(res["outline"], caption="Detected leaf outline")
c3.image(res["leaf_input"], caption="Input to the segmented model")

st.caption(f"Estimated lesion area: about {res['severity']:.1f}% of the leaf (rough estimate).")
st.caption(
    "Research prototype, not a diagnostic tool. It only knows three classes (healthy, "
    "angular leaf spot, bean rust) and expects a close-up photo of one leaf. Confidence "
    "scores can be overconfident."
)
