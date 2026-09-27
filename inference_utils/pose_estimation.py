import numpy as np
import cv2
from typing import Tuple


def _finite_alignment_mask(p_src: np.ndarray,
                           p_dst: np.ndarray,
                           conf_src: np.ndarray = None,
                           conf_dst: np.ndarray = None) -> np.ndarray:
    mask = np.isfinite(p_src).all(axis=1) & np.isfinite(p_dst).all(axis=1)
    if conf_src is not None:
        mask &= np.isfinite(conf_src)
    if conf_dst is not None:
        mask &= np.isfinite(conf_dst)
    return mask


def _safe_mean_l2(residuals: np.ndarray) -> float:
    distances = _safe_l2(residuals)
    valid = np.isfinite(distances)
    if not np.any(valid):
        return np.inf
    return float(np.mean(distances[valid]))


def _safe_l2(residuals: np.ndarray) -> np.ndarray:
    residuals = residuals.astype(np.float64, copy=False)
    distances = np.full(residuals.shape[0], np.inf, dtype=np.float64)
    valid = np.isfinite(residuals).all(axis=1)
    distances[valid] = np.hypot.reduce(residuals[valid], axis=1)
    return distances


def estimate_pose_from_2d3d(pred_target_aligned: np.ndarray,
                            mask: np.ndarray,
                            intrinsics: np.ndarray) -> np.ndarray:
    """
    Estimate object pose from 2D—3D correspondences using OpenCV's solvePnPRansac,
    with deterministic sampling and post-RANSAC pose refinement.

    Parameters:
    - pred_target_aligned: (H, W, 3) 3D model-space points (NOCS or aligned pointmap)
    - mask: (H, W) binary mask for valid pixels
    - intrinsics: (3, 3) camera intrinsic matrix

    Returns:
    - pose: (4, 4) model-to-camera pose matrix, or None if RANSAC fails
    """
    # ---- Deterministic RANSAC seed ----
    np.random.seed(42)
    cv2.setRNGSeed(42)

    # Collect 2D-3D correspondences
    ys, xs = np.where(mask > 0)
    if len(xs) < 6:
        print("Too few correspondences for PnP.")
        return np.eye(4, dtype=np.float32)

    image_points = np.stack([xs, ys], axis=1).astype(np.float32)
    object_points = pred_target_aligned[ys, xs].astype(np.float32)

    # RANSAC PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=intrinsics,
        distCoeffs=None,
        reprojectionError=10.0, # 1.0
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=400,
        confidence=0.99
    )

    if not success or inliers is None or len(inliers) < 6:
        print("PnP RANSAC failed or too few inliers.")
        return np.eye(4, dtype=np.float32)

    # Compose final 4x4 pose
    R, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = tvec.flatten()

    return pose

def weighted_umeyama(p_src: np.ndarray,
                     p_dst: np.ndarray,
                     conf_src: np.ndarray,
                     conf_dst: np.ndarray = None,
                     with_scaling: bool = True):
    finite_mask = _finite_alignment_mask(p_src, p_dst, conf_src, conf_dst)
    p_src = p_src[finite_mask].astype(np.float64, copy=False)
    p_dst = p_dst[finite_mask].astype(np.float64, copy=False)
    conf_src = conf_src[finite_mask].astype(np.float64, copy=False)
    if conf_dst is not None:
        conf_dst = conf_dst[finite_mask].astype(np.float64, copy=False)

    if p_src.shape[0] < 3:
        raise ValueError("Fewer than 3 finite point pairs, cannot align.")

    # 1. Compute pair weights.
    if conf_dst is None:
        w = conf_src.copy()
    else:
        w = conf_src * conf_dst

    # 2. Remove zero-weight pairs.
    mask = (w > 0)
    p_src_f = p_src[mask]
    p_dst_f = p_dst[mask]
    w = w[mask]

    if p_src_f.shape[0] < 3:
        raise ValueError("Fewer than 3 positive-weight point pairs, cannot align.")

    # 3. Normalize weights to sum to 1.
    w_sum = np.sum(w)
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("Invalid correspondence weights, cannot align.")
    w = w / w_sum

    # 4. Compute weighted centroids.
    mu_src = np.sum(p_src_f * w[:,None], axis=0)
    mu_dst = np.sum(p_dst_f * w[:,None], axis=0)

    # 5. Center the points.
    src_centered = p_src_f - mu_src
    dst_centered = p_dst_f - mu_dst

    # 6. Compute weighted covariance.
    #    Σ = ∑_i w_i · (dst_i - μ_dst) (src_i - μ_src)^T
    Sigma = dst_centered.T @ (src_centered * w[:,None])

    # 7. SVD.
    U, D, Vt = np.linalg.svd(Sigma)
    # Keep a right-handed coordinate system.
    S_mat = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S_mat[2,2] = -1
    R = U @ S_mat @ Vt

    # 8. Estimate scale.
    if with_scaling:
        # Weighted source variance: var_src = sum_i w_i ||src_centered_i||^2.
        var_src = np.sum(w * np.sum(src_centered**2, axis=1))
        if not np.isfinite(var_src) or var_src <= np.finfo(np.float64).eps:
            raise ValueError("Degenerate source point variance, cannot estimate scale.")
        # s = trace(D·S) / var_src
        s = np.trace(np.diag(D) @ S_mat) / var_src
    else:
        s = 1.0

    # 9. Compute translation.
    t = mu_dst - s * R @ mu_src

    return s, R, t

def robust_umeyama(p_src: np.ndarray,
                p_dst: np.ndarray,
                conf_src: np.ndarray,
                conf_dst: np.ndarray = None,
                with_scaling: bool = True,
                n_sample: int = 64, #128,
                iter_num: int = 100,
                eval_sample_size: int = 256, #1000,
                random_state: int = 42):
    """
    Robust weighted Umeyama via multi-sample and error-based selection.

    Args:
        p_src: (N, 3) source points
        p_dst: (N, 3) destination points
        conf_src: (N,) confidence of source points
        conf_dst: (N,) optional confidence of destination points
        with_scaling: whether to solve for scale
        n_sample: number of points per pose estimation
        iter_num: number of pose estimation iterations
        eval_sample_size: number of points used to evaluate error
        random_state: reproducibility seed

    Returns:
        s_best, R_best, t_best
    """
    finite_mask = _finite_alignment_mask(p_src, p_dst, conf_src, conf_dst)
    p_src = p_src[finite_mask].astype(np.float64, copy=False)
    p_dst = p_dst[finite_mask].astype(np.float64, copy=False)
    conf_src = conf_src[finite_mask]
    if conf_dst is not None:
        conf_dst = conf_dst[finite_mask]

    rng = np.random.RandomState(random_state)
    N = p_src.shape[0]
    if N < 3:
        return (1.0, np.eye(3), np.zeros(3))
    n_sample = min(n_sample, N)

    # For error evaluation, sample a fixed subset
    eval_idx = rng.choice(N, size=min(eval_sample_size, N), replace=False)
    p_src_eval = p_src[eval_idx]
    p_dst_eval = p_dst[eval_idx]

    best_err = np.inf
    best_pose = None

    for _ in range(iter_num):
        try:
            idx = rng.choice(N, size=n_sample, replace=False)
            s, R_, t = weighted_umeyama(
                p_src[idx], p_dst[idx],
                conf_src[idx],
                None if conf_dst is None else conf_dst[idx],
                with_scaling
            )

            # Compute mean error over downsampled eval points
            p_src_trans = (s * (R_ @ p_src_eval.T)).T + t
            err = _safe_mean_l2(p_dst_eval - p_src_trans)

            if err < best_err:
                best_err = err
                best_pose = (s, R_, t)

        except Exception:
            continue

    if best_pose is None:
        # raise RuntimeError("Failed to find a valid pose.")
        return (1.0, np.eye(3), np.zeros(3))  

    return best_pose
