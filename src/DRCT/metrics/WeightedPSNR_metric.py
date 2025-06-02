import torch
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def WeightedPSNR(denoised, ground_truth, max_pixel_value=255.0):
    psnr_values = []
    batch_size = denoised.size(0)

    for i in range(batch_size):
        mask = ground_truth[i] != 0
        mse = torch.sum((denoised[i] - ground_truth[i])[mask] ** 2) / mask.sum()
        psnr = 10 * torch.log10(max_pixel_value ** 2 / mse)
        psnr_values.append(psnr)
    return torch.stack(psnr_values).mean()
