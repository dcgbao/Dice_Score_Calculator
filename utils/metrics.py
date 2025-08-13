import numpy as np

def dice_score(pred, gt):
    """Compute Dice similarity coefficient."""
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt))
