import streamlit as st
import tempfile
from utils.io_utils import load_and_match
from utils.metrics import dice_score

st.title("Dice Score Calculator for nnU-Net Prediction vs Doctor Ground Truth")

pred_file = st.file_uploader("Upload Prediction (.nii.gz)", type=["nii.gz"])
gt_file = st.file_uploader("Upload Ground Truth (.nii.gz)", type=["nii.gz"])

if pred_file and gt_file:
    try:
        # Save temporary files so SimpleITK can read them
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_pred:
            tmp_pred.write(pred_file.read())
            pred_path = tmp_pred.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_gt:
            tmp_gt.write(gt_file.read())
            gt_path = tmp_gt.name

        # Load and compute Dice score
        pred_bin, gt_bin = load_and_match(pred_path, gt_path)
        score = dice_score(pred_bin, gt_bin)

        st.success(f"Dice Score: {score:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
