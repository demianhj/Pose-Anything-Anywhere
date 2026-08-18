import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def to_display_img(img, normalize=False):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.dtype == np.float16:
        img = img.astype(np.float32)

    if normalize:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
    return img

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _normalize_to_uint8(img, mask=None, cmap=None):
    """
    Normalize a 2D array to uint8 RGB:
    - If cmap is None: use grayscale.
    - Else: apply matplotlib cmap to [0,1].
    """
    H, W = img.shape
    # collect valid pixels for percentile clipping
    valid = img[mask > 0] if mask is not None else img.ravel()
    if valid.size == 0:
        base = np.zeros((H, W), dtype=np.uint8)
    else:
        vmin, vmax = np.percentile(valid, [1, 99])
        clipped = np.clip(img, vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin + 1e-6)
        if cmap is None:
            base = (norm * 255).astype(np.uint8)[..., None]
            return np.repeat(base, 3, axis=2)
        else:
            colored = cmap(norm)[..., :3]
            rgb = (colored * 255).astype(np.uint8)
            rgb[mask == 0] = 0
            return rgb
    return np.repeat(base[..., None], 3, axis=2)

def visualize_depth_err(depth_pred, depth_gt, mask):
    """
    Visualize the error between predicted and ground truth depth maps.
    The error is normalized and displayed as an RGB image.
    
    Parameters:
    - depth_pred: Predicted depth map (H, W)
    - depth_gt: Ground truth depth map (H, W)
    - mask: Binary mask indicating valid pixels (H, W)
    
    Returns:
    - RGB image of the error visualization
    """
    error = np.abs(depth_pred - depth_gt)
    valid = error[mask > 0]
    if valid.size == 0:
        H, W = error.shape
        return np.zeros((H, W, 3), dtype=np.uint8)
    vmin, vmax = np.percentile(valid, [1, 99])
    clipped = np.clip(error, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin + 1e-6)
    cmap_err = cm.get_cmap("coolwarm")
    rgb_err = cmap_err(norm)[..., :3]
    img_err = (rgb_err * 255).astype(np.uint8)
    img_err[mask == 0] = 0
    # get the avg error value
    avg_error = np.mean(valid)
    # add the avg error value to the image
    cv2.putText(img_err, f'Avg Error: {avg_error:.4f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img_err

def visualize_depth_res(rgb, depth_gt, depth_pred, mask):
    """
    Returns a uint8 array (H, 4W, 3) with columns:
     [ RGB | GT depth (plasma) | Pred depth (plasma) | Error (coolwarm) ]
    """
    rgb        = _to_numpy(rgb)
    depth_gt   = _to_numpy(depth_gt)
    depth_pred = _to_numpy(depth_pred)
    mask       = _to_numpy(mask)

    # Prepare RGB column
    if rgb.dtype != np.uint8:
        img_norm = np.clip(rgb, 0, 1) if rgb.max() <= 1 else rgb
        rgb_img = (img_norm * 255).astype(np.uint8)
    else:
        rgb_img = rgb

    # Use plasma colormap for depth maps
    plasma = cm.get_cmap("plasma")
    gt_img   = _normalize_to_uint8(depth_gt,   mask, cmap=plasma)
    pred_img = _normalize_to_uint8(depth_pred, mask, cmap=plasma)

    # Error heatmap
    err_img  = visualize_depth_err(depth_pred, depth_gt, mask)

    # Stitch together
    return np.concatenate([rgb_img, gt_img, pred_img, err_img], axis=1)

def visualize_pts(points_3d, mask=None, normalize=False, background_color=(0, 0, 0)):
    """
    Visualize 3D points (X,Y,Z) as an RGB image by mapping coordinates to RGB channels.
    Masked-out regions are set to a background color.

    Parameters:
    - points_3d: HxWx3 array of 3D points
    - mask: Optional HxW binary mask
    - background_color: Tuple of (R, G, B) in [0, 1] for background
    """

    H, W, _ = points_3d.shape

    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]

    def normalize(arr):
        valid = arr[mask > 0] if mask is not None else arr[arr > 0]
        min_val = valid.min()
        max_val = valid.max()
        return np.clip((arr - min_val) / (max_val - min_val + 1e-8), 0, 1)
    
    if normalize:
        x_norm = normalize(x)
        y_norm = normalize(y)
        z_norm = normalize(z)
        rgb_image = np.stack((x_norm, y_norm, z_norm), axis=-1)
    else:
        rgb_image = np.zeros((x, y, z), dtype=np.float32)

    if mask is not None:
        # Apply background color to areas where mask is 0
        mask_3d = mask[..., None] > 0
        rgb_image = np.where(mask_3d, rgb_image, background_color)

    # Convert to uint8
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

    # plt.imshow(rgb_image_uint8)
    # plt.show()

    return rgb_image_uint8

def visualize_RGB_pts(label, RGB, GT_pts, Pred_pts, dpi=100):
    """
    Create a side-by-side visualization of RGB and pts images and return it as a NumPy array,
    without using any save buffers, and with minimal white margins.
    """
    RGB = to_display_img(RGB, normalize=False)
    GT_pts = to_display_img(GT_pts, normalize=True)
    Pred_pts = to_display_img(Pred_pts, normalize=True)
    # Create figure and canvas
    fig = plt.figure(figsize=(10,6), dpi=dpi)
    canvas = FigureCanvas(fig)

    # Add axes manually to control position (left, bottom, width, height in [0, 1])
    ax1 = fig.add_axes([0.0, 0.0, 0.33, 1.0])
    ax2 = fig.add_axes([0.33, 0.0, 0.33, 1.0])
    ax3 = fig.add_axes([0.66, 0.0, 0.33, 1.0])

    ax1.imshow(RGB)
    ax1.axis('off')

    ax2.imshow(GT_pts)
    ax2.axis('off')

    ax3.imshow(Pred_pts)
    ax3.axis('off')

    # Set the title
    fig.suptitle(label, fontsize=16, y=0.95)

    # Draw and convert to NumPy array
    canvas.draw()
    width, height = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    vis_img = buf.reshape(height, width, 3)

    plt.close(fig)
    return vis_img


def axis_off_3d(ax):
    """Disable everything 3D axis-related, mimicking axis('off') for Axes3D."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)

    # Hide pane backgrounds
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Hide pane edges
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    # Hide 3D box lines
    try:
        ax.w_xaxis.line.set_color('white')
        ax.w_yaxis.line.set_color('white')
        ax.w_zaxis.line.set_color('white')
        ax.w_xaxis.line.set_linewidth(0.)
        ax.w_yaxis.line.set_linewidth(0.)
        ax.w_zaxis.line.set_linewidth(0.)
    except AttributeError:
        pass  # Safe fallback for newer versions

    ax.set_box_aspect([1, 1, 1])  # Maintain shape


def visualize_pts_in_3d(query_points_3d, ref_points_3d, max_points=4096):
    """
    Visualize two sets of 3D points: ref (red) and query (blue),
    with optional downsampling and return (N, 6) array: [X, Y, Z, R, G, B]
    """

    all_pts = []
    all_colors = []

    # --- Ref points: red ---
    if ref_points_3d is not None and len(ref_points_3d) > 0:
        ref_pts = ref_points_3d.astype(np.float32)
        ref_color = np.array([255, 0, 0], dtype=np.uint8)  # Red
        ref_colors = np.tile(ref_color[None, :], (ref_pts.shape[0], 1))
        all_pts.append(ref_pts)
        all_colors.append(ref_colors)

    # --- Query points: blue ---
    if query_points_3d is not None and len(query_points_3d) > 0:
        query_pts = query_points_3d.astype(np.float32)
        query_color = np.array([0, 0, 255], dtype=np.uint8)  # Blue
        query_colors = np.tile(query_color[None, :], (query_pts.shape[0], 1))
        all_pts.append(query_pts)
        all_colors.append(query_colors)

    # --- Combine ---
    pts_concat = np.concatenate(all_pts, axis=0)          # (N, 3)
    colors_concat = np.concatenate(all_colors, axis=0)    # (N, 3)
    pts_rgb = np.concatenate([pts_concat, colors_concat], axis=1)  # (N, 6)

    # --- Optional: downsample ---
    if pts_rgb.shape[0] > max_points:
        idx = np.random.choice(pts_rgb.shape[0], max_points, replace=False)
        pts_rgb = pts_rgb[idx]

    return pts_rgb  # shape: (N, 6), dtype: float32 for XYZ, uint8 for RGB

def pad_to_max_width(images, pad_value=10):
    max_width = max(img.shape[1] for img in images)
    padded_images = []

    for img in images:
        h, w, c = img.shape
        if w < max_width:
            pad_width = max_width - w
            padded = cv2.copyMakeBorder(img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
        else:
            padded = img
        padded_images.append(padded)
    
    return np.concatenate(padded_images, axis=0)

def visualize_matches(
    images,          # (S,3,H,W) torch.Tensor 或 np.ndarray
    query_xy,        # (N,2)  第0帧的像素坐标 (x,y)
    gt_xy,           # (S-1,N,2)  其余帧的GT像素坐标 (-1,-1 表示无匹配)
    pos_gt_idx,      # (S-1,N)  bool，是否有匹配
    positive_frames, # (S-1) 
    max_points=1369,  # 最多可视化的匹配条数（从有效样本里随机抽）
    radius=3,        # 画点半径
    seed=0,          # 随机种子
):
    """
    返回: canvas  (H, S*W, 3)  uint8
    - 左起第0块是 query image（frame-0）
    - 右侧依次是 frame-1, frame-2, ..., frame-(S-1)
    - 对每个有GT的 (0↔j) 画彩色点并连线
    """
    rng = np.random.RandomState(seed)

    # ---- 准备图像到 numpy uint8 ----
    if torch.is_tensor(images):
        if images.requires_grad: images = images.detach()
        images = images.cpu().numpy()
    imgs = np.asarray(images)
    if imgs.shape[1] == 3:  # (S,3,H,W) -> (S,H,W,3)
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    if imgs.dtype != np.uint8:
        mn, mx = float(imgs.min()), float(imgs.max())
        if mx <= mn + 1e-8:
            imgs = np.zeros_like(imgs, dtype=np.uint8)
        else:
            imgs = ((imgs - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    # 假定输入是 RGB，OpenCV 用 BGR 作画
    imgs_bgr = imgs[..., ::-1]
    S, H, W, _ = imgs_bgr.shape

    # ---- 画布拼接 ----
    canvas = np.concatenate([imgs_bgr[i] for i in range(S)], axis=1).copy()  # (H, S*W, 3)

    # ---- 坐标准备 ----
    if torch.is_tensor(query_xy):
        query_xy = query_xy.detach().cpu().numpy()
    if torch.is_tensor(gt_xy):
        gt_xy = gt_xy.detach().cpu().numpy()
    if torch.is_tensor(pos_gt_idx):
        pos_gt_idx = pos_gt_idx.detach().cpu().numpy().astype(bool)

    query_xy = np.round(query_xy).astype(int)        # (N,2)
    gt_xy = np.round(gt_xy).astype(int)           # (S-1,N,2)

    # 选择要显示的 query 索引（从所有 “至少在一帧有匹配” 的里采样）
    valid_any = pos_gt_idx.any(axis=0)               # (N,)
    ids = np.where(valid_any)[0]
    if ids.size == 0:
        return canvas[..., ::-1]  # 没有匹配，直接返回底图
    if ids.size > max_points:
        ids = rng.choice(ids, size=max_points, replace=False)

    # 为每个被选中的 query 分配一个颜色（BGR）
    colors = (rng.rand(len(ids), 3) * 255).astype(np.uint8)

    # ---- 绘制 ----
    for k, qi in enumerate(ids):
        c = tuple(int(v) for v in colors[k].tolist())

        x0, y0 = query_xy[qi]
        if 0 <= x0 < W and 0 <= y0 < H:
            cv2.circle(canvas, (x0, y0), radius, c, -1, lineType=cv2.LINE_AA)

        for j in range(1, S):
            if not pos_gt_idx[j-1, qi]:
                continue
            x1, y1 = gt_xy[j-1, qi]
            if not (0 <= x1 < W and 0 <= y1 < H):  # 越界保护
                continue
            x1_shift = x1 + j * W
            cv2.circle(canvas, (x1_shift, y1), radius, c, -1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x0, y0), (x1_shift, y1), c, 1, lineType=cv2.LINE_AA)
    
    # ---- 在 positive frame 顶部加标签 ----
    for j in range(1, S):
        if not positive_frames[j-1]:
            continue
        text = "positive view"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(canvas, text,
                    (j * W + (W - text_w) // 2, text_h + 10),
                    font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas[..., ::-1].copy()  # BGR->RGB

def visualize_correpondence(query_image, pred_images, filtered_query_xy, filtered_pred_xys, save_path=None):
    """
    Visualizes matched points between a query image and multiple prediction images,
    stacking the results vertically. It assumes the input points are already filtered.

    Args:
        query_image (np.ndarray): The original query image, either float (0-1) or
                                  uint8 (0-255). Expected shape is (H, W, 3).
        pred_images (np.ndarray): A collection of prediction images. Expected shape is
                                  (S, H, W, 3).
        filtered_query_xy (np.ndarray): Filtered coordinates from the query image.
                                        Shape is (N', 2).
        filtered_pred_xys (np.ndarray): Corresponding filtered coordinates from the
                                        prediction images. Shape is (S, N', 2).
        save_path (str, optional): An optional file path to save the output image.

    Returns:
        np.ndarray: The combined image with drawn lines, vertically stacked.
    """
    # 1. Prepare images for drawing by converting to uint8 and BGR
    def prepare_image(img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    query_image_display = prepare_image(query_image)
    
    list_of_combined_results = []
    S = pred_images.shape[0]
    
    # Determine the maximum width for consistent stacking
    max_w = max(query_image_display.shape[1], pred_images.shape[2])

    for s_idx in range(S):
        pred_image_s = prepare_image(pred_images[s_idx])
        pred_xy_s = filtered_pred_xys[s_idx]

        # 2. Filter out invalid matches (where pred_xy is [-1, -1])
        valid_indices = np.where(pred_xy_s[:, 0] != -1)[0]
        valid_query_xy = filtered_query_xy[valid_indices]
        valid_pred_xy = pred_xy_s[valid_indices]
        num_matches = len(valid_query_xy)

        # 3. Resize and combine images horizontally
        h_query, w_query, _ = query_image_display.shape
        h_pred, w_pred, _ = pred_image_s.shape
        
        q_resized = cv2.resize(query_image_display, (max_w, h_query), interpolation=cv2.INTER_AREA)
        p_resized = cv2.resize(pred_image_s, (max_w, h_pred), interpolation=cv2.INTER_AREA)
        
        combined_image = np.hstack((q_resized, p_resized))
        combined_image = np.ascontiguousarray(combined_image)

        # 4. Draw lines and annotate match count
        for i in range(num_matches):
            start_point = tuple(valid_query_xy[i].astype(int))
            end_point = (int(valid_pred_xy[i][0] + max_w), int(valid_pred_xy[i][1]))
            color = (int(np.random.randint(0, 256)), int(np.random.randint(0, 256)), int(np.random.randint(0, 256)))
            cv2.line(combined_image, start_point, end_point, color, 1)

        text = f"Matches: {num_matches}"
        text_pos = (max_w + 10, 30)
        cv2.putText(combined_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        list_of_combined_results.append(combined_image)
    
    # 5. Stack all combined images vertically
    if not list_of_combined_results:
        print("Warning: No matches found or no prediction images provided.")
        return np.zeros((1, max_w * 2, 3), dtype=np.uint8)
    
    final_output_image = np.vstack(list_of_combined_results)

    # 6. Save the final image if a path is provided
    if save_path:
        cv2.imwrite(save_path, final_output_image)

    return final_output_image