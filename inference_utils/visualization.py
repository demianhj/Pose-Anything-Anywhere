import numpy as np
import cv2
import open3d as o3d

def save_pointcloud(points, colors=None, save_path="pointcloud.ply"):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to .ply file
    o3d.io.write_point_cloud(save_path, pcd)

def open3d_vis(points, colors=None):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        # Make sure colors are in [0,1] float32 format
        colors = np.asarray(colors, dtype=np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud using Open3D's built-in GUI
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Open3D Point Cloud Viewer",
                                      width=800,
                                      height=600,
                                      point_show_normal=False)

def vis_pc(points):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Visualize the point cloud using Open3D's built-in GUI
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Open3D Point Cloud Viewer",
                                      width=800,
                                      height=600,
                                      point_show_normal=False)

def save_pc_together(pc1, pc2, save_path="pointcloud.ply"):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1.colors = o3d.utility.Vector3dVector(np.tile([[1, 0, 0]], (pc1.shape[0], 1)))  # Red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2.colors = o3d.utility.Vector3dVector(np.tile([[0, 1, 0]], (pc2.shape[0], 1)))  # Green

    pcd_combined = pcd1 + pcd2
    o3d.io.write_point_cloud(save_path, pcd_combined)

def vis_pc_together(pc1, pc2):
    # Create Open3D point cloud objects
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1.colors = o3d.utility.Vector3dVector(np.tile([[1, 0, 0]], (pc1.shape[0], 1)))  # Red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2.colors = o3d.utility.Vector3dVector(np.tile([[0, 1, 0]], (pc2.shape[0], 1)))  # Green

    # Visualize
    o3d.visualization.draw_geometries([pcd1, pcd2],
                                      window_name="Open3D Point Cloud Viewer",
                                      width=800,
                                      height=600,
                                      point_show_normal=False)

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img

def draw_text(image, text, position, font_scale=1, color=(255, 0, 0), thickness=2):
    # Use OpenCV's putText function to add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

def compute_ADD(model_points: np.ndarray,
                pred_pose: np.ndarray,
                gt_pose: np.ndarray) -> float:
    """
    Computes the ADD (Average Distance of Model Points) metric.

    Parameters:
    - model_points: (N, 3) array of 3D model points
    - pred_pose: (4, 4) predicted pose matrix
    - gt_pose: (4, 4) ground truth pose matrix

    Returns:
    - ADD error (float)
    """
    # Transform model points with each pose
    model_points_h = np.hstack([model_points, np.ones((model_points.shape[0], 1))])  # (N, 4)

    pred_transformed = (pred_pose @ model_points_h.T).T[:, :3]  # (N, 3)
    gt_transformed   = (gt_pose @ model_points_h.T).T[:, :3]    # (N, 3)

    # Compute average Euclidean distance
    add = np.mean(np.linalg.norm(pred_transformed - gt_transformed, axis=1))

    return add

def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def draw(img, axes, color, imgpts=None, axis_i=None):
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), color, 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), color, 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), color, 3)
    if axis_i is not None:
        img = cv2.line(img, tuple(axis_i[0]), tuple(axis_i[1]), (125, 0, 0), 3)
        img = cv2.line(img, tuple(axis_i[0]), tuple(axis_i[3]), (125, 125, 0), 3)
        img = cv2.line(img, tuple(axis_i[0]), tuple(axis_i[2]), (40, 60, 80), 3)

    return img

def draw_pose_from_axis(
    image,
    camera_K,
    camera_ex,
    camera_ex_gt,
    output_path,
    mask=None,
    padding_ratio=1.0,
    final_size=(384,384),
):
    """
    Draws predicted and ground-truth camera poses on an image.
    Optionally crops based on a mask, expands the crop by a padding ratio,
    adjusts the crop to match the final aspect ratio (to avoid distortion),
    and finally resizes to the target or original resolution.

    Args:
        image (np.ndarray): Input RGB image.
        camera_K (np.ndarray): Camera intrinsic matrix.
        camera_ex (np.ndarray): Estimated camera extrinsic matrix.
        camera_ex_gt (np.ndarray): Ground-truth camera extrinsic matrix.
        output_path (str): Path to save the visualization.
        mask (np.ndarray, optional): Binary mask for cropping (same HxW as image).
        padding_ratio (float, optional): Padding ratio relative to crop size (default 0.2).
        final_size (tuple[int, int], optional): Target resolution (W, H), e.g. (384, 384).
    """

    orig_h, orig_w = image.shape[:2]

    # Determine the target aspect ratio
    if final_size is not None:
        target_w, target_h = final_size
        target_aspect = target_w / target_h
    else:
        target_w, target_h = orig_w, orig_h
        target_aspect = orig_w / orig_h

    # === Step 1: Draw the projected axes ===
    xyz_axis = 0.1 * np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ]).T

    transformed_axes = transform_coordinates_3d(xyz_axis, camera_ex)
    projected_axes = calculate_2d_projections(transformed_axes, camera_K)

    transformed_axes_gt = transform_coordinates_3d(xyz_axis, camera_ex_gt)
    projected_axes_gt = calculate_2d_projections(transformed_axes_gt, camera_K)

    draw_image = draw(image, projected_axes_gt, (193, 182, 255))   # GT axes
    draw_image = draw(draw_image, projected_axes, (144, 238, 144)) # Predicted axes

    # === Step 2: Mask-based cropping ===
    if mask is not None:
        mask = mask.astype(bool)
        y_idx, x_idx = np.where(mask)

        if len(x_idx) > 0 and len(y_idx) > 0:
            # Tight bounding box around mask
            x_min, x_max = x_idx.min(), x_idx.max()
            y_min, y_max = y_idx.min(), y_idx.max()

            # Apply padding
            crop_w = x_max - x_min + 1
            crop_h = y_max - y_min + 1
            pad_w = int(crop_w * padding_ratio)
            pad_h = int(crop_h * padding_ratio)

            x_min -= pad_w
            x_max += pad_w
            y_min -= pad_h
            y_max += pad_h

            # === Step 3: Adjust crop box to match target aspect ratio ===
            crop_w = x_max - x_min + 1
            crop_h = y_max - y_min + 1
            crop_aspect = crop_w / crop_h if crop_h > 0 else target_aspect

            if crop_aspect > target_aspect:
                # Crop is too wide → expand height
                new_h = int(crop_w / target_aspect)
                delta_h = (new_h - crop_h) // 2
                y_min -= delta_h
                y_max += delta_h
            else:
                # Crop is too tall → expand width
                new_w = int(crop_h * target_aspect)
                delta_w = (new_w - crop_w) // 2
                x_min -= delta_w
                x_max += delta_w

            # Clip to valid image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_w - 1, x_max)
            y_max = min(orig_h - 1, y_max)

            # Validate box
            if x_max > x_min and y_max > y_min:
                draw_image = draw_image[y_min:y_max+1, x_min:x_max+1]
            else:
                print("Invalid crop box after adjustment. Skipping crop.")
                draw_image = image.copy()
        else:
            print("No valid mask region found. Skipping crop.")

    # === Step 4: Resize to target resolution ===
    draw_image = cv2.resize(draw_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # === Step 5: Save result (RGB→BGR for cv2) ===
    cv2.imwrite(output_path, draw_image[:, :, ::-1].astype(np.uint8))

        
        


        

