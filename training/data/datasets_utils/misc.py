import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import training.data.datasets_utils.structs as structs
from training.data.datasets_utils.structs import AlignedBox2f, CameraModel, PinholePlaneCameraModel
from typing import Any, Optional, Tuple, Tuple, TypeVar

AnyTensor = TypeVar("AnyTensor", np.ndarray, "torch.Tensor")

def get_rigid_matrix(trans: structs.RigidTransform) -> np.ndarray:
    """Creates a 4x4 transformation matrix from a 3x3 rotation and 3x1 translation.

    Args:
        trans: A rigid transformation defined by a 3x3 rotation matrix and
            a 3x1 translation vector.
    Returns:
        A 4x4 rigid transformation matrix.
    """

    matrix = np.eye(4)
    matrix[:3, :3] = trans.R
    matrix[:3, 3:] = trans.t
    return matrix

def get_bbox(mask: np.ndarray) -> AlignedBox2f:
    """Get the bounding box of a binary mask.

    Args:
        mask: Binary mask.
    Returns:
        Bounding box of the mask.
    """

    if not np.any(mask):
        return AlignedBox2f(0, 0, 0, 0)

    y_indices, x_indices = np.nonzero(mask)
    return AlignedBox2f(
        left=np.min(x_indices),
        top=np.min(y_indices),
        right=np.max(x_indices) + 1,
        bottom=np.max(y_indices) + 1,
    )

def calc_crop_box(
    box: AlignedBox2f,
    box_scaling_factor: float = 1.0,
    make_square: bool = False,
) -> AlignedBox2f:
    """Adjusts a bounding box to the specified aspect and scale.

    Args:
        box: Bounding box.
        box_aspect: The aspect ratio of the target box.
        box_scaling_factor: The scaling factor to apply to the box.
    Returns:
        Adjusted box.
    """

    # Potentially inflate the box and adjust aspect ratio.
    crop_box_width = box.width * box_scaling_factor
    crop_box_height = box.height * box_scaling_factor

    # Optionally make the box square.
    if make_square:
        crop_box_side = max(crop_box_width, crop_box_height)
        crop_box_width = crop_box_side
        crop_box_height = crop_box_side

    # Calculate padding.
    x_pad = 0.5 * (crop_box_width - box.width)
    y_pad = 0.5 * (crop_box_height - box.height)

    return AlignedBox2f(
        left=box.left - x_pad,
        top=box.top - y_pad,
        right=box.right + x_pad,
        bottom=box.bottom + y_pad,
    )

def construct_crop_camera(
    box: AlignedBox2f,
    camera_model_c2w: CameraModel,
    viewport_size: Tuple[int, int],
    viewport_rel_pad: float,
) -> CameraModel:
    """Constructs a virtual pinhole camera from the specified 2D bounding box.

    Args:
        camera_model_c2w: Original camera model with extrinsics set to the
            camera->world transformation.

        viewport_crop_size: Viewport size of the new camera.
        viewport_scaling_factor: Requested scaling of the viewport.
    Returns:
        A virtual pinhole camera whose optical axis passes through the center
        of the specified 2D bounding box and whose focal length is set such as
        the sphere representing the bounding box (+ requested padding) is visible
        in the camera viewport.
    """

    # Get centroid and radius of the reference sphere (the virtual camera will
    # be constructed such as the projection of the sphere fits the viewport.
    f = 0.5 * (camera_model_c2w.f[0] + camera_model_c2w.f[1])
    cx, cy = camera_model_c2w.c
    box_corners_in_c = np.array(
        [
            [box.left - cx, box.top - cy, f],
            [box.right - cx, box.top - cy, f],
            [box.left - cx, box.bottom - cy, f],
            [box.right - cx, box.bottom - cy, f],
        ]
    )
    box_corners_in_c /= np.linalg.norm(box_corners_in_c, axis=1, keepdims=True)
    centroid_in_c = np.mean(box_corners_in_c, axis=0)
    centroid_in_c_h = np.hstack([centroid_in_c, 1]).reshape((4, 1))
    centroid_in_w = camera_model_c2w.T_world_from_eye.dot(centroid_in_c_h)[:3, 0]

    radius = np.linalg.norm(box_corners_in_c - centroid_in_c, axis=1).max()

    # Transformations from world to the original and virtual cameras.
    trans_w2c = np.linalg.inv(camera_model_c2w.T_world_from_eye)
    trans_w2vc = gen_look_at_matrix(trans_w2c, centroid_in_w)

    # Transform the centroid from world to the virtual camera.
    centroid_in_vc = transform_3d_points_numpy(
        trans_w2vc, np.expand_dims(centroid_in_w, axis=0)
    ).squeeze()

    # Project the sphere radius to the image plane of the virtual camera and
    # enlarge it by the specified padding. This defines the 2D extent that
    # should be visible in the virtual camera.
    fx_fy_orig = np.array(camera_model_c2w.f, dtype=np.float32)
    radius_2d = fx_fy_orig * radius / centroid_in_vc[2]
    extent_2d = (1.0 + viewport_rel_pad) * radius_2d

    cx_cy = np.array(viewport_size, dtype=np.float32) / 2.0 - 0.5

    # Set the focal length such as all projected points fit the viewport of the
    # virtual camera.
    fx_fy = fx_fy_orig * cx_cy / extent_2d

    # Parameters of the virtual camera.
    return PinholePlaneCameraModel(
        width=viewport_size[0],
        height=viewport_size[1],
        f=tuple(fx_fy),
        c=tuple(cx_cy),
        T_world_from_eye=np.linalg.inv(trans_w2vc),
    )

def get_intrinsic_matrix(cam: CameraModel) -> np.ndarray:
    """Returns a 3x3 intrinsic matrix of the given camera.

    Args:
        cam: The input camera model.
    Returns:
        A 3x3 intrinsic matrix K.
    """

    return np.array(
        [
            [cam.f[0], 0.0, cam.c[0]],
            [0.0, cam.f[1], cam.c[1]],
            [0.0, 0.0, 1.0],
        ]
    )

def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: Optional[Any] = None,
) -> np.ndarray:
    """Resizes an image.

    Args:
      image: An input image.
      size: The size of the output image (width, height).
      interpolation: An interpolation method (a suitable one is picked if undefined).
    Returns:
      The resized image.
    """

    if interpolation is None:
        interpolation = (
            cv2.INTER_AREA if image.shape[0] >= size[1] else cv2.INTER_LINEAR
        )
    return cv2.resize(image, size, interpolation=interpolation)

def crop_image(image: np.ndarray, crop_box: AlignedBox2f) -> np.ndarray:
    """Crops an image.

    Args:
        image: The input HWC image.
        crop_box: The bounding box for cropping given by (x1, y1, x2, y2).
    Returns:
        Cropped image.
    """

    return image[crop_box.top : crop_box.bottom, crop_box.left : crop_box.right]

def warp_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> np.ndarray:
    """
    Warp an image from the source camera to the destination camera.

    Parameters
    ----------
    src_camera :
        Source camera model
    dst_camera :
        Destination camera model
    src_image :
        Source image
    interpolation :
        Interpolation method
    depth_check :
        If True, mask out points with negative z coordinates
    factor_to_downsample :
        If this value is greater than 1, it will downsample the input image prior to warping.
        This improves downsampling performance, in an attempt to replicate
        area interpolation for crop+undistortion warps.
    """

    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)

def warp_depth_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_depth_image: np.ndarray,
    depth_check: bool = True,
) -> np.ndarray:

    # Copy the source depth image.
    depth_image = np.array(src_depth_image)

    # If the camera extrinsics changed, update the depth values.
    if not np.allclose(src_camera.T_world_from_eye, dst_camera.T_world_from_eye):

        # Image coordinates with valid depth values.
        valid_mask = depth_image > 0
        ys, xs = np.nonzero(valid_mask)

        # Transform the source depth image to a point cloud.
        pts_in_src = src_camera.window_to_eye(np.vstack([xs, ys]).T)
        pts_in_src *= np.expand_dims(depth_image[valid_mask] / pts_in_src[:, 2], axis=1)

        # Transform the point cloud from the source to the target camera.
        pts_in_w = src_camera.eye_to_world(pts_in_src)
        pts_in_trg = dst_camera.world_to_eye(pts_in_w)

        depth_image[valid_mask] = pts_in_trg[:, 2]

    # Warp the depth image to the target camera.
    return warp_image(
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_image=depth_image,
        interpolation=cv2.INTER_NEAREST,
        depth_check=depth_check,
    )

def transform_3d_points_numpy(trans: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform 3D points. Compute trans * points

    Args:
        points: 3D points of shape (num_points, 3).
        trans: Transformation matrix of shape (4, 4).
    Returns:
        Transformed 3D points of shape (num_points, 3).
    """

    assert trans.shape == (4, 4)
    assert points.shape[1] == 3
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    return trans.dot(points_h.T)[:3, :].T

def gen_look_at_matrix(
    orig_camera_from_world: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
    return_camera_from_world: bool = True,
) -> np.ndarray:
    """
    Rotates the input camera such that the new transformation align the z-direction to the provided point in world.
    Args:
      camera_angle is used to apply a roll rotation around the new z
      return_camera_from_world is used to return the inverse

    Returns:
        world_from_aligned_camera or aligned_camera_from_world
    """

    center_local = transform_points(orig_camera_from_world, center)
    z_dir_local = center_local / np.linalg.norm(center_local)
    delta_r_local = from_two_vectors(
        np.array([0, 0, 1], dtype=center.dtype), z_dir_local
    )
    orig_world_from_camera = np.linalg.inv(orig_camera_from_world)

    world_from_aligned_camera = orig_world_from_camera.copy()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ delta_r_local
    )

    # Locally rotate the z axis to align with the camera angle
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ z_local_rot
    )

    if return_camera_from_world:
        return np.linalg.inv(world_from_aligned_camera)
    return world_from_aligned_camera

def from_two_vectors(a_orig: np.ndarray, b_orig: np.ndarray) -> np.ndarray:
    # Convert the vectors to unit vectors.
    a = normalized(a_orig)
    b = normalized(b_orig)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = skew_matrix(v)

    rot = (
        np.eye(3, 3, dtype=a_orig.dtype)
        + v_mat
        + np.matmul(v_mat, v_mat) * (1 - c) / (max(s * s, 1e-15))
    )

    return rot

def skew_matrix(v: np.ndarray) -> np.ndarray:
    res = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype
    )
    return res

def normalized(v: AnyTensor, axis: int = -1, eps: float = 5.43e-20) -> AnyTensor:
    """
    Return a unit-length copy of vector(s) v

    Parameters
    ----------
    axis : int = -1
        Which axis to normalize on

    eps
        Epsilon to avoid division by zero. Vectors with length below
        eps will not be normalized. The default is 2^-64, which is
        where squared single-precision floats will start to lose
        precision.
    """
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d

def transform_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Transform an array of 3D points with an SE3 transform (rotation and translation).

    *WARNING* this function does not support arbitrary affine transforms that also scale
    the coordinates (i.e., if a 4x4 matrix is provided as input, the last row of the
    matrix must be `[0, 0, 0, 1]`).

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points [..., 3]

    Returns:
        Transformed points [..., 3]
    """
    return rotate_points(matrix, points) + matrix[..., :3, 3]

def rotate_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Rotates an array of 3D points with an affine transform,
    which is equivalent to transforming an array of 3D rays.

    *WARNING* This ignores the translation in `m`; to transform 3D *points*, use
    `transform_points()` instead.

    Note that we specifically optimize for ndim=2, which is a frequent
    use case, for better performance. See n388920 for the comparison.

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points or 3d direction vectors [..., 3]

    Returns:
        Rotated points / direction vectors [..., 3]
    """
    if matrix.ndim == 2:
        return (points.reshape(-1, 3) @ matrix[:3, :3].T).reshape(points.shape)
    else:
        return (matrix[..., :3, :3] @ points[..., None]).squeeze(-1)

def as_4x4(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Append [0,0,0,1] to convert 3x4 matrices to a 4x4 homogeneous matrices

    If the matrices are already 4x4 they will be returned unchanged.
    """
    if a.shape[-2:] == (4, 4):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (3, 4):
        return np.concatenate(
            (
                a,
                np.broadcast_to(
                    np.array([0, 0, 0, 1], dtype=a.dtype), a.shape[:-2] + (1, 4)
                ),
            ),
            axis=-2,
        )
    raise ValueError("expected 3x4 or 4x4 affine transform")


