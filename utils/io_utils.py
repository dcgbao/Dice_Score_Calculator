import SimpleITK as sitk
import numpy as np

def load_and_match(pred_path, gt_path):
    """Load prediction & ground truth NIfTI and resample pred to GT space."""
    pred_img = sitk.ReadImage(pred_path)
    gt_img = sitk.ReadImage(gt_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(gt_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    pred_resampled = resampler.Execute(pred_img)

    pred_array = sitk.GetArrayFromImage(pred_resampled)
    gt_array = sitk.GetArrayFromImage(gt_img)

    return (pred_array > 0).astype(np.uint8), (gt_array > 0).astype(np.uint8)