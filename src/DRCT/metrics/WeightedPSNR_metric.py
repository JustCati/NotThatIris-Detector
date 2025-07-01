import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def WeightedPSNR(img, img2, max_pixel_value=255.0, **kwargs):
    mask = img2 != 0
    mse = np.sum((img - img2)[mask] ** 2) / mask.sum()
    psnr = 10 * np.log10(max_pixel_value ** 2 / mse)
    return psnr
