import torch

def heatmaps_to_coords(heatmaps):
    """
    Преобразует heatmaps (B, K, H, W) в координаты (B, K, 2) — (x, y)
    """
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)  # (B, K, H*W)
    max_inds = heatmaps_reshaped.argmax(dim=2)   # (B, K)

    # Преобразуем в координаты
    coords = torch.zeros(B, K, 2, device=heatmaps.device)
    coords[..., 0] = (max_inds % W).float()  # x
    coords[..., 1] = (max_inds // W).float()  # y
    return coords

def rescale_coords(coords, heatmap_size, image_size):
    h, w = heatmap_size
    H, W = image_size
    scale_x = W / w
    scale_y = H / h
    coords[..., 0] *= scale_x
    coords[..., 1] *= scale_y
    return coords

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Обратно нормализует тензор изображения [C, H, W]"""
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return tensor * std + mean