import torch
import torch.nn as nn

from vggt.mv_match.utils.model_utils import pairwise_distance
from vggt.mv_match.utils.vis_utils import visualize_gt_corresp, visualize_correspondences

def get_gt_corresp(points, gt_r, gt_t):
    """
    Transform points from camera coordinates to world/object coordinates using rotation and translation.
    
    Args:
        points (torch.Tensor): Point cloud of shape (b, n, 3) on CUDA.
        gt_r (torch.Tensor): Rotation matrices of shape (b, 3, 3) on CUDA.
        gt_t (torch.Tensor): Translation vectors of shape (b, 3) on CUDA.
    
    Returns:
        torch.Tensor: Transformed points in world/object coordinates of shape (b, n, 3).
    """
    batch_size, num_points, _ = points.shape
    device = points.device
    
    # Construct extrinsic matrices (b, 4, 4)
    ext = torch.zeros((batch_size, 4, 4), device=device)
    ext[:, :3, :3] = gt_r
    ext[:, :3, 3] = gt_t
    ext[:, 3, 3] = 1.0
    
    # Compute inverse of extrinsic matrix
    ext_inv = torch.inverse(ext)
    
    # Convert points to homogeneous coordinates
    ones = torch.ones((batch_size, num_points, 1), device=device)
    points_homogeneous = torch.cat([points, ones], dim=2)  # Shape: (b, n, 4)
    
    # Reshape for batch matrix multiplication
    points_homogeneous = points_homogeneous.transpose(1, 2)  # Shape: (b, 4, n)
    
    # Apply transformation
    transformed_points = torch.bmm(ext_inv, points_homogeneous)  # Shape: (b, 4, n)
    
    # Convert back from homogeneous coordinates
    transformed_points = transformed_points.transpose(1, 2)  # Shape: (b, n, 4)
    transformed_points = transformed_points[:, :, :3]  # Shape: (b, n, 3)
    
    return transformed_points


def compute_rotation_error(R_est, R_gt):
    """Compute rotation error with batched inputs using PyTorch.
    
    Args:
        R_est: Estimated rotation matrices of shape (batch_size, 3, 3) on GPU.
        R_gt: Ground-truth rotation matrices of shape (batch_size, 3, 3) on GPU.
        
    Returns:
        Rotation errors in degrees of shape (batch_size,) on GPU.
    """
    # Compute the relative rotation: R = R_est @ R_gt.T
    R_rel = torch.bmm(R_est, R_gt.transpose(1, 2))
    
    # Compute the trace of the relative rotation matrices
    trace = torch.diagonal(R_rel, dim1=1, dim2=2).sum(dim=1)
    
    # Clamp the trace values to valid range for arccos
    trace = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    
    # Compute the rotation angle in radians and convert to degrees
    angles_rad = torch.acos(trace)
    angles_deg = angles_rad * 180.0 / torch.pi
    
    return angles_deg


def compute_translation_errors(t_est, t_gt):
    """Compute translation error with batched inputs using PyTorch.
    
    Args:
        t_est: Estimated translation vectors of shape (batch_size, 3) on GPU.
        t_gt: Ground-truth translation vectors of shape (batch_size, 3) on GPU.
        
    Returns:
        Translation errors (Euclidean distance) of shape (batch_size,) on GPU.
    """
    # Compute the absolute difference
    err = torch.abs(t_est - t_gt)
    
    # Compute the Euclidean norm along the last axis
    return torch.norm(err, dim=-1)


def transform_3d_points_torch(R, t, points):
    """Transform 3D points with batched inputs using PyTorch.
    
    Args:
        R: Rotation matrices of shape (batch_size, 3, 3) on GPU.
        t: Translation vectors of shape (batch_size, 3) on GPU.
        points: 3D points of shape (batch_size, num_points, 3) on GPU.
        
    Returns:
        Transformed 3D points of shape (batch_size, num_points, 3) on GPU.
    """
    batch_size = R.shape[0]
    
    # Apply rotation to each point
    # (batch_size, 3, 3) @ (batch_size, 3, num_points) = (batch_size, 3, num_points)
    rotated_points = torch.bmm(R, points.transpose(1, 2))
    
    # Add translation to each point
    # Reshape t from (batch_size, 3) to (batch_size, 3, 1)
    translated_points = rotated_points + t.view(batch_size, 3, 1)
    
    # Transpose back to get (batch_size, num_points, 3)
    transformed_points = translated_points.transpose(1, 2)
    
    return transformed_points


def compute_pose_loss(end_points, gt_R, gt_t, loss_str='pose'):
    """Compute pose loss using PyTorch operations on GPU.
    
    Args:
        end_points: Dictionary containing model endpoints.
        pred_R: Predicted rotation matrices of shape (batch_size, 3, 3) on GPU.
        pred_t: Predicted translation vectors of shape (batch_size, 3) on GPU.
        gt_R: Ground-truth rotation matrices of shape (batch_size, 3, 3) on GPU.
        gt_t: Ground-truth translation vectors of shape (batch_size, 3) on GPU.
        loss_str: String prefix for storing results in end_points dict.
        
    Returns:
        Updated end_points dictionary with error metrics.
    """
    pred_R = end_points['pred_R']
    pred_t = end_points['pred_t']

    vertices_in_gt = transform_3d_points_torch(gt_R, gt_t, end_points['pts'])
    vertices_in_pred = transform_3d_points_torch(pred_R, pred_t, end_points['pts'])
    
    # Compute point-wise errors
    point_errors = torch.sqrt(torch.sum((vertices_in_gt - vertices_in_pred) ** 2, dim=-1))
    
    # Compute rotation and translation errors
    rotation_errors = compute_rotation_error(pred_R, gt_R)
    translation_errors = compute_translation_errors(pred_t * 1000, gt_t * 1000)
    
    # Store results in end_points
    end_points[loss_str + '_point_error'] = point_errors.mean()
    end_points[loss_str + '_rotation_error'] = rotation_errors.mean()
    end_points[loss_str + '_translation_error'] = translation_errors.mean()    
    
    return end_points

def compute_correspondence_loss(
    atten_list,
    pts1,
    pts2,
    dis_thres,
):
    CE = nn.CrossEntropyLoss(reduction ='none')
    # visualize_gt_corresp(pts1[0].detach().cpu().numpy(), pts2[0].detach().cpu().numpy()) # check if the gt_pts are correct
    dis_mat = torch.sqrt(pairwise_distance(pts1, pts2)) # pts1: query pts2: target

    dis1, label1 = dis_mat.min(2)
    fg_label1 = (dis1<=dis_thres).float()
    label1 = (fg_label1 * (label1.float()+1.0)).long()

    dis2, label2 = dis_mat.min(1)
    fg_label2 = (dis2<=dis_thres).float()
    label2 = (fg_label2 * (label2.float()+1.0)).long()

    # loss
    loss = []
    for idx, atten in enumerate(atten_list):
        l1 = CE(atten.transpose(1,2)[:,:,1:].contiguous(), label1).mean(1)
        l2 = CE(atten[:,:,1:].contiguous(), label2).mean(1)
        loss_idx =  0.5 * (l1 + l2)
        loss.append(loss_idx)
    loss = torch.clamp(torch.stack(loss, dim=1).mean(1), max=100.0)

    # # visualization
    # # Convert to numpy for visualization (for the first item in the batch)
    # pts1_np = pts1[0].detach().cpu().numpy()
    # pts2_np = pts2[0].detach().cpu().numpy()
    # label1_np = label1[0].detach().cpu().numpy()
    # visualize_correspondences(pts1_np, pts2_np, label1_np)

    # get the matching pred
    pred_label = torch.max(atten_list[-1][:,1:,:], dim=2)[1]
    pred_acc = (pred_label==label1).float().mean(1)
    gt_label = label1

    return loss, pred_label, gt_label


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, end_points):
        out_dicts = {'loss': 0}
        for key in end_points.keys():
            if 'fine_' in key or 'coarse_' in key:
                out_dicts[key] = end_points[key].mean()
                if 'loss' in key:
                    out_dicts['loss'] = out_dicts['loss'] + end_points[key]
        if isinstance(out_dicts['loss'], torch.Tensor): # check if it's training
            out_dicts['loss'] = torch.clamp(out_dicts['loss'], max=100.0).mean()
        end_points.update({"loss": out_dicts['loss']})
        return out_dicts


class depth_Loss(nn.Module):
    def __init__(self):
        super(depth_Loss, self).__init__()

    def forward(self, end_points):
        out_dicts = {'loss': 0}
        for key in end_points.keys():
            if 'depth_loss' in key:
                out_dicts[key] = end_points[key].mean()
                out_dicts['loss'] = out_dicts['loss'] + end_points[key]
        if isinstance(out_dicts['loss'], torch.Tensor): # check if it's training
            out_dicts['loss'] = torch.clamp(out_dicts['loss'], max=100.0).mean()
        end_points.update({"loss": out_dicts['loss']})
        return out_dicts

