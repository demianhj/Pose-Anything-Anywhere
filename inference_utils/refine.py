import numpy as np
import torch

def get_correspondence_torch(    pred_query, gt_refer, 
    k_nn=5, device="cuda", 
    max_batch=10000,   # Predicted points per batch.
    subsample_gt=10000 # Optional GT sample count.
):
    """
    GPU-friendly mutual nearest-neighbor matching.
    - Supports batched processing to avoid OOM.
    - Optional GT sampling.
    """

    H, W, _ = pred_query.shape
    flat_pts = torch.tensor(pred_query.reshape(-1, 3), device=device, dtype=torch.float32)
    mask_flat = torch.any(flat_pts != 0, dim=1)
    flat_ids = torch.nonzero(mask_flat).squeeze(1)
    pts_pred = flat_pts[flat_ids]
    pts_gt = torch.tensor(gt_refer, device=device, dtype=torch.float32)

    if pts_pred.numel() == 0 or pts_gt.numel() == 0:
        return (np.empty((0, 3)), np.empty((0, 3)), np.zeros((H, W), dtype=bool))

    # Optional sampling.
    if subsample_gt is not None and pts_gt.shape[0] > subsample_gt:
        idx = torch.randperm(pts_gt.shape[0], device=device)[:subsample_gt]
        pts_gt = pts_gt[idx]

    N_pred, N_gt = pts_pred.shape[0], pts_gt.shape[0]

    # ====== 1️⃣ batched KNN: pred → gt ======
    idx_forward_list = []
    for start in range(0, N_pred, max_batch):
        end = min(start + max_batch, N_pred)
        dists = torch.cdist(pts_pred[start:end], pts_gt)  # (B, N_gt)
        _, idx = torch.topk(dists, k_nn, largest=False, dim=1)
        idx_forward_list.append(idx)
        del dists
        torch.cuda.empty_cache()
    idx_forward = torch.cat(idx_forward_list, dim=0)

    # ====== 2️⃣ batched KNN: gt → pred ======
    idx_backward_list = []
    for start in range(0, N_gt, max_batch):
        end = min(start + max_batch, N_gt)
        dists = torch.cdist(pts_gt[start:end], pts_pred)  # (B, N_pred)
        _, idx = torch.topk(dists, k_nn, largest=False, dim=1)
        idx_backward_list.append(idx)
        del dists
        torch.cuda.empty_cache()
    idx_backward = torch.cat(idx_backward_list, dim=0)

    # ====== 3️⃣ mutual check ======
    mutual_mask = torch.zeros(N_pred, dtype=torch.bool, device=device)
    mutual_model_idx = torch.full((N_pred,), -1, dtype=torch.long, device=device)

    gt_to_pred = torch.zeros(N_gt, N_pred, dtype=torch.bool, device=device)
    gt_indices = torch.arange(N_gt, device=device).unsqueeze(1).expand(-1, k_nn)
    gt_to_pred[gt_indices, idx_backward] = True

    pred_indices = torch.arange(N_pred, device=device).unsqueeze(1).expand(-1, k_nn)
    mutual_links = gt_to_pred[idx_forward, pred_indices]  # (N_pred, k)
    has_mutual = mutual_links.any(dim=1)
    first_match = torch.argmax(mutual_links.float(), dim=1)
    mutual_model_idx[has_mutual] = idx_forward[torch.arange(N_pred, device=device), first_match][has_mutual]
    mutual_mask = has_mutual

    # Return results.
    pts_pred_mutual = pts_pred[mutual_mask].cpu().numpy()
    pts_model_mutual = pts_gt[mutual_model_idx[mutual_mask]].cpu().numpy()

    out = np.zeros_like(pred_query)
    keep_flat_ids = flat_ids[mutual_mask].cpu().numpy()
    ys, xs = np.unravel_index(keep_flat_ids, (H, W))
    out[ys, xs] = pts_model_mutual
    query_mask = np.any(out != 0, axis=-1)

    return pts_pred_mutual, pts_model_mutual, query_mask
