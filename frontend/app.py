"""Upload a food image to the inference service."""

import os

import requests
import streamlit as st


INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Food-11 classifier", page_icon="🍽️")
st.title("Food-11 classifier")
st.write("Upload a food photo to predict its category.")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, width=300)
    if st.button("Classify image", type="primary"):
        try:
            with st.spinner("Classifying…"):
                response = requests.post(
                    f"{INFERENCE_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=(5, 60),
                )
                response.raise_for_status()
                result = response.json()
            st.success(f"Prediction: {result['category']} ({result['confidence']:.1%})")
        except requests.HTTPError:
            st.error("The image could not be classified. Please try another JPG or PNG.")
        except requests.RequestException:
            st.error("The prediction service is unavailable. Please try again shortly.")
        except (ValueError, KeyError, TypeError):
            st.error("The prediction service returned an unexpected response.")
