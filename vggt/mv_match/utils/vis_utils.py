import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import random
import cv2
import torch
from sklearn.decomposition import PCA
import torch.nn.functional as F

def vis_gt_matching(image1, image2, sampled_choose_1, sampled_choose_2, gt_label, max_points=100):
    valid_mask = (gt_label > 0)
    source_positions = torch.where(valid_mask.squeeze(0))[0]      
    indices_1_pts = sampled_choose_1[source_positions]
    target_positions = gt_label[0, source_positions] - 1
    indices_2_pts = sampled_choose_2[target_positions]
    coords1_2d_gt = convert_indices_to_2d_coords(indices_1_pts.cpu().numpy(), image_width=518)
    coords2_2d_gt = convert_indices_to_2d_coords(indices_2_pts.cpu().numpy(), image_width=518)
    if len(coords1_2d_gt) > max_points:
        selected_indices = np.random.choice(len(coords1_2d_gt), max_points, replace=False)
        coords1_2d_gt = coords1_2d_gt[selected_indices]
        coords2_2d_gt = coords2_2d_gt[selected_indices]
    vis_gt_match = visualize_2d_matches(image1, image2, coords1_2d_gt, coords2_2d_gt)
    return vis_gt_match

# def vis_pred_matching(image1, image2, sampled_choose_1, sampled_choose_2, pred_label, gt_label=None, max_points=100):
#     if gt_label is None:
#         valid_mask = (pred_label > 0)
#     else:
#         valid_mask = (pred_label == gt_label) & (gt_label > 0)
#     source_positions = torch.where(valid_mask.squeeze(0))[0]
#     indices_1_pts = sampled_choose_1[source_positions]
#     target_positions = pred_label[0, source_positions] - 1
#     indices_2_pts = sampled_choose_2[target_positions]
#     coords1_2d_pred = convert_indices_to_2d_coords(indices_1_pts.cpu().numpy(), image_width=518)
#     coords2_2d_pred = convert_indices_to_2d_coords(indices_2_pts.cpu().numpy(), image_width=518)
#     if valid_mask.sum() > max_points:
#         selected_indices = np.random.choice(len(coords1_2d_pred), max_points, replace=False)
#         coords1_2d_pred = coords1_2d_pred[selected_indices]
#         coords2_2d_pred = coords2_2d_pred[selected_indices]
#     vis_pred_match = visualize_2d_matches(image1, image2, coords1_2d_pred, coords2_2d_pred)
#     if gt_label is not None:
#         cv2.putText(vis_pred_match, f'Correct Predictions: {valid_mask.sum()}/{(gt_label > 0).sum()}', 
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     return vis_pred_match

def vis_pred_matching(image1, image2, sampled_choose_1, sampled_choose_2, pred_label, gt_label, max_points=100):
    """
    Visualizes predicted feature point matches, color-coding correct and incorrect ones,
    and prioritizing the display of correct matches.

    Args:
        image1 (np.ndarray): The first image (RGB, Grayscale, or BGR).
        image2 (np.ndarray): The second image (RGB, Grayscale, or BGR).
        sampled_choose_1 (torch.Tensor): Original 1D indices of sampled points in image 1.
        sampled_choose_2 (torch.Tensor): Original 1D indices of sampled points in image 2.
        pred_label (torch.Tensor): The predicted match labels.
        gt_label (torch.Tensor): The ground truth match labels.
        max_points (int): The maximum number of match lines to draw.

    Returns:
        np.ndarray: A RGB image containing the visualization result.
    """
    # Dynamically get the image width to avoid hardcoding
    h, w, _ = image1.shape
    
    # Define colors (in RGB format for OpenCV)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # 1. Identify correct and incorrect matches
    correct_mask = (pred_label == gt_label) & (gt_label > 0)
    incorrect_mask = (pred_label != gt_label) & (pred_label > 0)

    # Helper function: extract 2D coordinates based on a mask
    def get_coords_from_mask(mask):
        if mask.sum() == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        
        source_pos = torch.where(mask.squeeze(0))[0]
        indices_1 = sampled_choose_1[source_pos]
        target_pos = pred_label[0, source_pos] - 1
        indices_2 = sampled_choose_2[target_pos]
        
        # This is an external function for converting 1D indices to 2D coordinates.
        # You need to ensure this function is available.
        coords1 = convert_indices_to_2d_coords(indices_1.cpu().numpy(), image_width=w)
        coords2 = convert_indices_to_2d_coords(indices_2.cpu().numpy(), image_width=w)
        return coords1, coords2

    coords1_correct, coords2_correct = get_coords_from_mask(correct_mask)
    coords1_incorrect, coords2_incorrect = get_coords_from_mask(incorrect_mask)

    # 2. Sampling logic: prioritize keeping correct matches
    num_correct = len(coords1_correct)
    num_incorrect = len(coords1_incorrect)

    if num_correct >= max_points:
        # If there are enough correct matches, sample only from them
        indices = np.random.choice(num_correct, max_points, replace=False)
        final_coords1_correct = coords1_correct[indices]
        final_coords2_correct = coords2_correct[indices]
        final_coords1_incorrect = np.empty((0, 2))
        final_coords2_incorrect = np.empty((0, 2))
    else:
        # Keep all correct matches
        final_coords1_correct = coords1_correct
        final_coords2_correct = coords2_correct
        
        # And sample from incorrect matches to fill up to max_points
        num_to_sample_incorrect = min(max_points - num_correct, num_incorrect)
        if num_to_sample_incorrect > 0:
            indices = np.random.choice(num_incorrect, num_to_sample_incorrect, replace=False)
            final_coords1_incorrect = coords1_incorrect[indices]
            final_coords2_incorrect = coords2_incorrect[indices]
        else:
            final_coords1_incorrect = np.empty((0, 2))
            final_coords2_incorrect = np.empty((0, 2))

    # 3. Drawing logic
    # Create the canvas
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    # Place the images onto the canvas
    for i, img in enumerate([image1, image2]):
        # If image is grayscale, convert to RGB so it has 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)        
        h_img, w_img = img.shape[:2]
        if i == 0:
            # Place the first image on the left
            canvas[:h_img, :w_img] = img
        else:
            # Place the second image on the right
            canvas[:h_img, w1:] = img

    # Helper function: draw match lines on the canvas
    def draw_lines(coords1, coords2, color):
        for i in range(len(coords1)):
            pt1 = (int(coords1[i][0]), int(coords1[i][1]))
            pt2 = (int(coords2[i][0] + w1), int(coords2[i][1]))
            cv2.line(canvas, pt1, pt2, color, 1)
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)

    # Draw incorrect matches (red) first, then correct ones (green) on top
    draw_lines(final_coords1_incorrect, final_coords2_incorrect, RED)
    draw_lines(final_coords1_correct, final_coords2_correct, GREEN)

    # 4. Add statistical information text
    tp = correct_mask.sum().item()
    fp = incorrect_mask.sum().item()
    total_gt = (gt_label > 0).sum().item()
    total_pred = tp + fp

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    
    text_counts = f"TP: {tp}, FP: {fp}, GT: {total_gt}"
    text_acc = f"Precision: {precision:.2%} and Recall: {recall:.2%}"

    cv2.putText(canvas, text_counts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(canvas, text_acc, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    
    return canvas

def vis_matching_res(image1, image2, sampled_choose_1, sampled_choose_2, pred_label, pred_conf, max_points=100):
    """
    Visualizes predicted feature point matches, prioritizing high-confidence matches.

    Args:
        image1 (np.ndarray): First image, H×W×3 or H×W (uint8 or float).
        image2 (np.ndarray): Second image, same format as image1.
        sampled_choose_1 (np.ndarray): 1D indices of sampled points in image 1.
        sampled_choose_2 (np.ndarray): 1D indices of sampled points in image 2.
        pred_label (np.ndarray): Predicted match labels (>0 means valid match, 0 means invalid).
        pred_conf (np.ndarray): Confidence scores of shape (1, N) or (N,).
        max_points (int): Max number of lines to draw.

    Returns:
        np.ndarray: RGB visualization canvas.
    """

    h, w = image1.shape[:2]

    # --- helper: convert flat indices to (x, y) coordinates ---
    def convert_indices_to_2d_coords(indices, image_width):
        ys = indices // image_width
        xs = indices % image_width
        return np.stack([xs, ys], axis=-1)

    # --- 1. 有效匹配筛选 ---
    pred_label = np.asarray(pred_label)
    pred_conf = np.asarray(pred_conf)
    valid_mask = (pred_label[0] > 0)
    valid_indices = np.flatnonzero(valid_mask)

    target_pos = pred_label[0, valid_indices].astype(np.int64)

    ok = (target_pos >= 0) & (target_pos < sampled_choose_2.shape[0])
    valid_indices = valid_indices[ok]
    target_pos = target_pos[ok]

    # 对应的置信度
    conf_valid = pred_conf[0, valid_indices] if pred_conf.ndim == 2 else pred_conf[valid_indices]

    # --- 2. 根据置信度排序，取前 max_points ---
    if len(conf_valid) > 0:
        sorted_idx = np.argsort(-conf_valid)  # 从高到低
        top_idx = sorted_idx[:min(max_points, len(sorted_idx))]
        valid_indices = valid_indices[top_idx]
        target_pos = target_pos[top_idx]
        conf_valid = conf_valid[top_idx]
    else:
        print("⚠️ No valid matches found.")
        return np.hstack([image1, image2])

    # --- 3. 转为坐标 ---
    indices_1 = sampled_choose_1[valid_indices]
    indices_2 = sampled_choose_2[target_pos]

    coords1 = convert_indices_to_2d_coords(indices_1, image_width=w)
    coords2 = convert_indices_to_2d_coords(indices_2, image_width=w)

    # --- 4. 创建画布 ---
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    def to_rgb(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] != 3:
            raise ValueError("Unsupported image shape")
        img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        return img

    image1 = to_rgb(image1)
    image2 = to_rgb(image2)

    canvas[:h1, :w1] = image1
    canvas[:h2, w1:w1 + w2] = image2

    # --- 5. 绘制匹配线条 ---
    # 置信度映射为颜色（高→绿，低→红）
    min_conf, max_conf = conf_valid.min(), conf_valid.max()
    conf_norm = (conf_valid - min_conf) / (max_conf - min_conf + 1e-8)

    for i in range(len(coords1)):
        pt1 = tuple(coords1[i].astype(int))
        pt2 = (int(coords2[i, 0] + w1), int(coords2[i, 1]))
        # 从红到绿渐变
        color = (
            int(255 * (1 - conf_norm[i])),  # R
            int(255 * conf_norm[i]),        # G
            0                               # B
        )
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)

    return canvas

def visualize_2d_matches(image1, image2, points1_2d, points2_2d, point_radius=3, line_thickness=1):
    """
    Visualizes 2D matches using only OpenCV.
    Draws lines between two images placed side-by-side and returns the result.

    Args:
        image1 (np.ndarray): The first image (in BGR or Grayscale format).
        image2 (np.ndarray): The second image (in BGR or Grayscale format).
        points1_2d (np.ndarray): Keypoints in the first image, shape (K, 2).
        points2_2d (np.ndarray): Corresponding keypoints in the second image, shape (K, 2).
        point_radius (int, optional): Radius of the circles drawn at keypoints. Defaults to 5.
        line_thickness (int, optional): Thickness of lines and circle outlines. Defaults to 2.

    Returns:
        np.ndarray: A BGR image of the visualization in NumPy array format.
    """
    # Ensure the number of points matches
    assert points1_2d.shape[0] == points2_2d.shape[0], "Number of points in both sets must be equal."
    
    # Get image dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create a new wide canvas to place both images side-by-side
    combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    # If images are grayscale, convert them to BGR to be able to draw color
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
    # Place the images onto the canvas
    combined_image[:h1, :w1, :] = image1
    combined_image[:h2, w1:w1 + w2, :] = image2
    
    # Draw lines and circles for each match
    for i in range(points1_2d.shape[0]):
        # Get coordinates for image 1 (and convert them to integer tuples)
        pt1 = (int(points1_2d[i, 0]), int(points1_2d[i, 1]))
        
        # Get coordinates for image 2 and shift them by the width of image 1
        pt2 = (int(points2_2d[i, 0] + w1), int(points2_2d[i, 1]))
        
        # Draw the line connecting the two points
        line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(combined_image, pt1, pt2, color=line_color, thickness=line_thickness)
        
        # Draw a circle at each keypoint
        cv2.circle(combined_image, pt1, radius=point_radius, color=line_color, thickness=line_thickness)
        cv2.circle(combined_image, pt2, radius=point_radius, color=line_color, thickness=line_thickness)
        
    return combined_image

def convert_indices_to_2d_coords(indices, image_width):
    """
    Converts flattened 1D indices back to 2D image coordinates (x, y).

    Args:
        indices (np.ndarray or list): A list or array of 1D indices.
        image_width (int): The width (w) of the corresponding 2D image.

    Returns:
        np.ndarray: An array of 2D coordinates in (x, y) format, shape (N, 2).
    """
    # Ensure the input is a numpy array
    indices = np.asarray(indices) 
    
    # y = index // width (row = index integer division width)
    coords_y = indices // image_width
    
    # x = index % width (column = index modulo width)
    coords_x = indices % image_width
    
    # Stack x and y coordinates together and transpose to get (N, 2) shape
    return np.vstack((coords_x, coords_y)).T

def visualize_correspondences(pts1, pts2, gt_label_with_offset, offset=None):
    """
    Visualizes correspondences, adding a translation offset to the second 
    point cloud for clarity.

    Args:
        pts1 (np.ndarray): The first point cloud, shape (N, 3).
        pts2 (np.ndarray): The second point cloud, shape (M, 3).
        gt_label_with_offset (np.ndarray): Correspondence indices that have been 
                                           processed with +1 and zeroing. Shape (N,).
        offset (np.ndarray, optional): A (3,) vector to translate the second point cloud.
                                       If None, a suitable offset is calculated automatically.
    """
    # --- 1. Calculate and apply the offset ---
    # Create a copy of pts2 for visualization to avoid modifying the original data
    pts2_display = np.copy(pts2)

    if offset is None:
        # If no offset is provided, calculate one automatically
        # Calculate the span of the first point cloud on the X-axis
        x_max = np.max(pts1[:, 0])
        x_min = np.min(pts1[:, 0])
        x_span = x_max - x_min
        # Translate the second point cloud by 1.2 times the span on the X-axis to display them side-by-side
        auto_offset = np.array([x_span * 1.2, 0, 0])
        pts2_display += auto_offset
        print(f"Offset not provided, automatically calculated: {auto_offset}")
    else:
        # Use the user-provided offset
        pts2_display += offset

    # --- 2. Create Open3D visualization objects ---
    # pcd1 uses original coordinates
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.paint_uniform_color([1, 0.7, 0])  # Orange

    # pcd2 uses the new coordinates with the offset
    pcd2_display = o3d.geometry.PointCloud()
    pcd2_display.points = o3d.utility.Vector3dVector(pts2_display)
    pcd2_display.paint_uniform_color([0, 0.6, 1])  # Blue

    # --- 3. Correct indices and create lines ---
    # 1. Find the indices of all valid matches (i.e., where label > 0)
    #    These are the indices of the points in pts1
    valid_indices_in_pts1 = np.where(gt_label_with_offset > 0)[0]

    # 2. Get the corresponding indices in pts2 for these matches (still with the +1 offset)
    indices_in_pts2_offset = gt_label_with_offset[valid_indices_in_pts1]

    # 3. Subtract 1 from the indices to restore the correct original indices in pts2
    correct_indices_in_pts2 = indices_in_pts2_offset - 1
    
    # Create the correspondence list [[idx_in_pcd1, idx_in_pcd2], ...]
    correspondences = np.asarray([[i, j] for i, j in zip(valid_indices_in_pts1, correct_indices_in_pts2)])

    # --- 4. Visualization ---
    geometries_to_draw = [pcd1, pcd2_display]
    if len(correspondences) > 0:
        # Note: We need to pass pcd1 and the new pcd2_display here
        line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pcd1, pcd2_display, correspondences
        )
        line_set.paint_uniform_color([0, 1, 0]) # Green lines
        geometries_to_draw.append(line_set)
        print(f"Successfully visualized {len(correspondences)} correspondences.")
    else:
        print("No valid correspondences found to visualize.")

    o3d.visualization.draw_geometries(geometries_to_draw, window_name="Correspondences Visualization")

def visualize_gt_corresp(whole_pts, mesh, save_path='debug_vis', image_name="visualization.png"):
    """
    Visualize 3D points and mesh together using matplotlib and save as a 2D image.
    
    Args:
        whole_pts (numpy.ndarray): Point cloud of shape (H, W, 3).
        mesh (trimesh.Trimesh): 3D mesh object.
        save_path (str): Directory to save the visualization.
        image_name (str): Name of the output image file.
        obj_id (int, optional): Object ID for the title.
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Process the point cloud
    valid_pts = whole_pts.reshape(-1, 3)
    valid_mask = ~np.isnan(valid_pts).any(axis=1) & ~np.isinf(valid_pts).any(axis=1)
    valid_mask = valid_mask & np.any(valid_pts != 0, axis=1)
    valid_pts = valid_pts[valid_mask]
    
    # Randomly sample points if there are too many (for performance)
    if valid_pts.shape[0] > 5000:
        indices = np.random.choice(valid_pts.shape[0], 5000, replace=False)
        valid_pts = valid_pts[indices]
    
    # Scatter plot for the points
    ax.scatter(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], 
            c='r', marker='.', s=1, alpha=0.5)
    
    # Add the mesh
    vertices = mesh
    
    # Plot mesh 
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               alpha=0.3, color='blue')
    
    # Add a text annotation instead of a legend
    ax.text2D(0.05, 0.95, "Red: Point Cloud\nBlue: Mesh", transform=ax.transAxes)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Get the limits for all axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    # Calculate the range for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    # Find the greatest range to ensure equal scaling
    max_range = max(x_range, y_range, z_range)
    
    # Set new limits based on the center of the original limits
    x_center = (x_limits[1] + x_limits[0]) / 2
    y_center = (y_limits[1] + y_limits[0]) / 2
    z_center = (z_limits[1] + z_limits[0]) / 2
    
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
    # Create the output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, image_name), dpi=300, bbox_inches='tight')
    plt.close(fig)

def vis_feat(pts1, pts2, feat1, feat2):
    col_1, col_2 = pca_features(feat1, feat2)
    fig = visualize_feat_pca(pts1, pts2, col_1, col_2)
    return fig

def pca_features(feat_1, feat_2):
    """
    Combines two feature sets, applies PCA, normalizes the result, 
    and then splits them back.
    """
    n_components = 3
    
    # Store the number of points in the first feature set to split them later
    n_points_1 = feat_1.shape[0]
    
    # 1. Concatenate the features *before* doing anything else
    combined_feat = np.vstack([feat_1, feat_2])
    
    # 2. Apply PCA on the entire combined dataset
    pca = PCA(n_components=n_components)
    feat_transformed = pca.fit_transform(combined_feat)
    
    # 3. Normalize each component across the combined data
    # A small epsilon (1e-8) is added for numerical stability to avoid division by zero
    feat_min = feat_transformed.min(axis=0)
    feat_max = feat_transformed.max(axis=0)
    feat_transformed = (feat_transformed - feat_min) / (feat_max - feat_min + 1e-8)
    
    # 4. Split the normalized data back into two separate sets
    feat1_transformed = feat_transformed[:n_points_1]
    feat2_transformed = feat_transformed[n_points_1:]
    
    return feat1_transformed, feat2_transformed

def visualize_feat_pca(pts1, pts2, col_pts1, col_pts2, canvas_size=(800, 800), view='xy', point_size=4, padding=0.2):
    """
    Visualizes two point clouds side-by-side, each centered in its own half 
    of the canvas to maximize detail and ensure they are centered.
    """
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    width, height = canvas_size
    panel_width = width // 2

    # Select axes for projection based on the view
    if view == 'xy':
        h_axis, v_axis = 0, 1
    elif view == 'xz':
        h_axis, v_axis = 0, 2
    else: # 'yz'
        h_axis, v_axis = 1, 2

    # --- NORMALIZE AND SCALE ---
    # 1. Calculate the ranges (size) of each point cloud individually
    pts1_h_range = pts1[:, h_axis].max() - pts1[:, h_axis].min()
    pts1_v_range = pts1[:, v_axis].max() - pts1[:, v_axis].min()
    pts2_h_range = pts2[:, h_axis].max() - pts2[:, h_axis].min()
    pts2_v_range = pts2[:, v_axis].max() - pts2[:, v_axis].min()
    
    # 2. Find the single largest dimension across BOTH clouds
    # This ensures both are drawn at the same scale.
    max_range = max(pts1_h_range, pts1_v_range, pts2_h_range, pts2_v_range)
    
    # Add padding to the scale
    scale_factor = (1 + padding) * max_range
    if scale_factor == 0: scale_factor = 1 # Avoid division by zero

    # --- DRAW EACH POINT CLOUD IN ITS PANEL ---
    all_pts = [pts1, pts2]
    all_cols = [col_pts1, col_pts2]
    
    for i, (pts, colors) in enumerate(zip(all_pts, all_cols)):
        # Calculate the center of the current point cloud
        center_h = (pts[:, h_axis].min() + pts[:, h_axis].max()) / 2
        center_v = (pts[:, v_axis].min() + pts[:, v_axis].max()) / 2
        
        # Determine the center of the panel (left or right)
        panel_center_x = panel_width // 2 + i * panel_width

        for point_3d, color_rgb in zip(pts, colors):
            h_coord = point_3d[h_axis]
            v_coord = point_3d[v_axis]

            # Normalize coordinates relative to the cloud's center and the global scale
            norm_h = (h_coord - center_h) / scale_factor
            norm_v = (v_coord - center_v) / scale_factor

            # Map normalized coordinates to pixel locations within the correct panel
            px = int(norm_h * panel_width + panel_center_x)
            py = int(-norm_v * panel_width + height / 2) # Use panel_width for v too, for square pixels

            # Draw the point if it's within the canvas bounds
            if 0 <= px < width and 0 <= py < height:
                color_bgr = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))
                cv2.circle(canvas, (px, py), radius=point_size, color=color_bgr, thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"View: {view.upper()}", (10, 30), font, 1, (0, 0, 0), 2)

    return canvas

# def visualize_feat_pca(pts1, pts2, col_pts1, col_pts2, x_offset=5.0):
#     """
#     Args:
#         pts1 (numpy): Source point cloud (n1, 3).
#         pts2 (numpy): Target point cloud (n2, 3).
#     """
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     for ax in fig.axes:
#         ax.grid(True)
    
#     # Plot points
#     ax.scatter(pts1[:, 0] + x_offset, pts1[:, 1], pts1[:, 2], c=col_pts1, label='pts1')
#     ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c=col_pts2, label='pts2')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     ax.set_xlim(-1, 6)
#     ax.set_ylim(-3.5, 3.5)
#     ax.set_zlim(-3.5, 3.5)
#     ax.legend()

#     return fig