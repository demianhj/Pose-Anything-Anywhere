import numpy as np
import cv2
from typing import Tuple

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

def scale_and_align(src: np.ndarray, dst: np.ndarray):
    """
    Compute the optimal non‑negative scale s that best maps src→dst (in a 
    least‑squares sense), and return both s and the aligned source cloud.
    
    Args:
      src: (N,3) source points
      dst: (N,3) target points (corresponding to src)
    
    Returns:
      s:   float scale factor ≥ 0
      src_aligned: (N,3) = s*(src - centroid_src) + centroid_dst
    """
    # 1) compute centroids
    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)

    # 2) center the clouds
    src_cent = src - src_center
    dst_cent = dst - dst_center

    # 3) least‑squares scale (guaranteed ≥0 if both clouds are roughly positively correlated)
    num = np.sum(src_cent * dst_cent)
    den = np.sum(src_cent**2)
    s_raw = num / den
    s = max(0.0, s_raw)

    return s, src_center, dst_center

def umeyama_alignment(p_src: np.ndarray, p_dst: np.ndarray, with_scaling=True):
    # p_src, p_dst: shape (N,3), 已知对应点
    N = p_src.shape[0]
    # 1. 质心
    mu_src = p_src.mean(axis=0)
    mu_dst = p_dst.mean(axis=0)
    # 2. 中心化
    src_centered = p_src - mu_src
    dst_centered = p_dst - mu_dst
    # 3. 计算协方差
    Sigma = dst_centered.T @ src_centered / N
    # 4. SVD
    # U, D, Vt = np.linalg.svd(Sigma)
    try:
        U, D, Vt = np.linalg.svd(Sigma)
    except np.linalg.LinAlgError as e:
        # 这里可以降级为不带缩放，或直接返回单位变换
        raise RuntimeError(f"SVD 未收敛: {e}")
    
    # 确保右手坐标系
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2,2] = -1
    R = U @ S @ Vt
    # 5. 计算尺度
    if with_scaling:
        var_src = np.sum(src_centered**2) / N
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    # 6. 平移
    t = mu_dst - s * R @ mu_src
    # 返回变换
    return s, R, t

def weighted_umeyama(p_src: np.ndarray,
                     p_dst: np.ndarray,
                     conf_src: np.ndarray,
                     conf_dst: np.ndarray = None,
                     with_scaling: bool = True):
    # 1. 计算每对点的权重
    if conf_dst is None:
        w = conf_src.copy()
    else:
        w = conf_src * conf_dst

    # 2. 去除权重为 0 的对
    mask = (w > 0)
    p_src_f = p_src[mask]
    p_dst_f = p_dst[mask]
    w = w[mask]

    # 3. 归一化权重使其和为 1
    w = w / np.sum(w)

    # 4. 计算加权质心
    mu_src = np.sum(p_src_f * w[:,None], axis=0)
    mu_dst = np.sum(p_dst_f * w[:,None], axis=0)

    # 5. 去中心化
    src_centered = p_src_f - mu_src
    dst_centered = p_dst_f - mu_dst

    # 6. 计算加权协方差矩阵
    #    Σ = ∑_i w_i · (dst_i - μ_dst) (src_i - μ_src)^T
    Sigma = dst_centered.T @ (src_centered * w[:,None])

    # 7. SVD 分解
    U, D, Vt = np.linalg.svd(Sigma)
    # 保证右手系
    S_mat = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S_mat[2,2] = -1
    R = U @ S_mat @ Vt

    # 8. 估计尺度
    if with_scaling:
        # 加权源方差：var_src = ∑_i w_i ||src_centered_i||^2
        var_src = np.sum(w * np.sum(src_centered**2, axis=1))
        # s = trace(D·S) / var_src
        s = np.trace(np.diag(D) @ S_mat) / var_src
    else:
        s = 1.0

    # 9. 计算平移
    t = mu_dst - s * R @ mu_src

    return s, R, t

def ransac_weighted_umeyama(p_src: np.ndarray,
                            p_dst: np.ndarray,
                            conf_src: np.ndarray,
                            conf_dst: np.ndarray = None,
                            with_scaling: bool = True,
                            iter_num: int = 1000,
                            inlier_thresh: float = 1, #1,
                            random_state: int = 42):
    """
    RANSAC-based weighted Umeyama alignment:
      - iter_num:        maximum number of RANSAC iterations
      - inlier_thresh:   distance threshold for inliers (same unit as point coordinates)

    Returns:
      best scale s, rotation R, translation t, and inlier mask
    """
    # Total number of points
    N = p_src.shape[0]
    rng = np.random.RandomState(random_state)
    best_inlier_wsum = -np.inf
    best_model = None
    best_inlier_mask = None

    # Combine weights for all points
    if conf_dst is None:
        w_all = conf_src.copy()
    else:
        w_all = conf_src * conf_dst

    # Use only points with positive weight for sampling
    valid_idx = np.where(w_all > 0)[0]
    if valid_idx.size < 3:
        raise ValueError("Fewer than 3 valid points, cannot perform RANSAC.")

    for _ in range(iter_num):
        # 1. Randomly sample 3 point indices
        samp = rng.choice(valid_idx, size=3, replace=False)
        ps = p_src[samp]
        pd = p_dst[samp]
        cs = conf_src[samp]
        cd = conf_dst[samp] if conf_dst is not None else None

        # 2. Fit weighted Umeyama on the sample
        try:
            s_cand, R_cand, t_cand = weighted_umeyama(ps, pd, cs, cd, with_scaling)
        except Exception:
            continue

        # 3. Compute reprojection errors for all points
        p_src_trans = (s_cand * (R_cand @ p_src.T)).T + t_cand
        errs = np.linalg.norm(p_dst - p_src_trans, axis=1)

        # 4. Identify inliers: error < threshold and positive weight
        inlier_mask = (errs < inlier_thresh) & (w_all > 0)

        # 5. Select model with highest total inlier weight
        wsum = np.sum(w_all[inlier_mask])
        if wsum > best_inlier_wsum:
            best_inlier_wsum = wsum
            best_model = (s_cand, R_cand, t_cand)
            best_inlier_mask = inlier_mask

    if best_model is None:
        raise RuntimeError("RANSAC failed to find a valid model. Consider adjusting parameters.")

    # 6. Refit weighted Umeyama using the best inliers
    inliers = best_inlier_mask
    s_final, R_final, t_final = weighted_umeyama(
        p_src[inliers], p_dst[inliers],
        conf_src[inliers],
        None if conf_dst is None else conf_dst[inliers],
        with_scaling
    )

    return s_final, R_final, t_final, best_inlier_mask

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
    rng = np.random.RandomState(random_state)
    N = p_src.shape[0]

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
            err = np.mean(np.linalg.norm(p_dst_eval - p_src_trans, axis=1))

            if err < best_err:
                best_err = err
                best_pose = (s, R_, t)

        except Exception:
            continue

    if best_pose is None:
        # raise RuntimeError("Failed to find a valid pose.")
        return (1.0, np.eye(3), np.zeros(3))  

    return best_pose