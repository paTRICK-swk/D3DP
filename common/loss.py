from matplotlib.pyplot import bone
import torch
import numpy as np
import torch.nn as nn

def mpjpe(predicted, target, return_joints_err=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if not return_joints_err:
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    else:
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        # errors: [B, T, N]
        from einops import rearrange
        errors = rearrange(errors, 'B T N -> N (B T)')
        errors = torch.mean(errors, dim=-1).cpu().numpy().reshape(-1) * 1000
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1)), errors

def mpjpe_diffusion_all_min(predicted, target, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    #assert predicted.shape == target.shape
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        # print(predicted.shape)
        # print(target.shape)
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        from einops import rearrange
        #errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
        errors = rearrange(errors, 'b t h f n  -> t h b f n', )
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        min_errors = min_errors.reshape(t, -1)
        min_errors = torch.mean(min_errors, dim=-1, keepdim=False)
        return min_errors
    else:
        t = predicted.shape[1]
        h = predicted.shape[2]
        mean_pose = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
        errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors

def mpjpe_diffusion_reproj(predicted, target, reproj_2d, target_2d):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    #assert predicted.shape == target.shape

    t = predicted.shape[1]
    h = predicted.shape[2]
    # print(predicted.shape)
    # print(target.shape)
    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    errors = torch.norm(predicted - target, dim=len(target.shape)-1)  # b,t,h,f,n
    errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1)
    from einops import rearrange
    #errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
    select_ind = torch.min(errors_2d, dim=2, keepdim=True).indices# b,t,1,f,n
    errors_select = torch.gather(errors, 2, select_ind)# b,t,1,f,n
    errors_select = rearrange(errors_select, 'b t h f n  -> t h b f n', )
    errors_select = errors_select.reshape(t, -1)
    errors_select = torch.mean(errors_select, dim=-1, keepdim=False)
    return errors_select

def mpjpe_diffusion(predicted, target, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    #assert predicted.shape == target.shape
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        # print(predicted.shape)
        # print(target.shape)
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        from einops import rearrange
        #errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
        errors = rearrange(errors, 'b t h f n  -> t h b f n', ).reshape(t, h, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        return min_errors
    else:
        t = predicted.shape[1]
        h = predicted.shape[2]
        mean_pose = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
        errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors

def mpjpe_diffusion_3dhp(predicted, target, valid_frame, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    #assert predicted.shape == target.shape
    from einops import rearrange

    valid_frame = valid_frame.squeeze(2)
    predicted = rearrange(predicted, 'b t h f n c  -> b f t h n c', )
    predicted_valid = predicted[valid_frame] # f, t, h, n, c
    target_valid = target[valid_frame] # f, n, c

    if not mean_pos:
        t = predicted_valid.shape[1]
        h = predicted_valid.shape[2]
        # print(predicted.shape)
        # print(target.shape)
        target_valid = target_valid.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1)
        errors = torch.norm(predicted_valid - target_valid, dim=len(target_valid.shape)-1)
        from einops import rearrange
        #errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
        errors = rearrange(errors, 'f t h n  -> t h f n', ).reshape(t, h, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        return min_errors
    else:
        t = predicted_valid.shape[1]
        h = predicted_valid.shape[2]
        mean_pose = torch.mean(predicted_valid, dim=2, keepdim=False)
        target_valid = target_valid.unsqueeze(1).repeat(1, t, 1, 1)
        errors = torch.norm(mean_pose - target_valid, dim=len(target_valid.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'f t n -> t f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))


def p_mpjpe_diffusion_all_min(predicted, target, mean_pos=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    #assert predicted.shape == target.shape


    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    if not mean_pos:
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    else:
        predicted = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t_sz, 1, 1, 1)

    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    if not mean_pos:
        target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        # from einops import rearrange
        # # errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
        # errors = rearrange(errors, 'b t h f n  -> t h b f n', )
        errors = errors.transpose(1, 2, 0, 3, 4) # t, h, b, f, n
        min_errors = np.min(errors, axis=1, keepdims=False)
        min_errors = min_errors.reshape(t_sz, -1)
        min_errors = np.mean(min_errors, axis=1, keepdims=False)
        return min_errors
    else:
        target = target.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        # from einops import rearrange
        # errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.transpose(1, 0, 2, 3)
        errors = errors.reshape(t_sz, -1)
        errors = np.mean(errors, axis=1, keepdims=False)
        return errors

def p_mpjpe_diffusion(predicted, target, mean_pos=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    #assert predicted.shape == target.shape


    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    if not mean_pos:
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    else:
        predicted = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t_sz, 1, 1, 1)

    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    if not mean_pos:
        target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        # from einops import rearrange
        # # errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
        # errors = rearrange(errors, 'b t h f n  -> t h b f n', )
        errors = errors.transpose(1, 2, 0, 3, 4).reshape(t_sz, h_sz, -1) # t, h, b, f, n
        errors = np.mean(errors, axis=2, keepdims=False)
        min_errors = np.min(errors, axis=1, keepdims=False)
        return min_errors
    else:
        target = target.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        # from einops import rearrange
        # errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.transpose(1, 0, 2, 3)
        errors = errors.reshape(t_sz, -1)
        errors = np.mean(errors, axis=1, keepdims=False)
        return errors

def p_mpjpe_diffusion_reproj(predicted, target, reproj_2d, target_2d):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    #assert predicted.shape == target.shape

    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape

    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1) # b, t, h, f, n
    selec_ind = torch.min(errors_2d, dim=2, keepdims=True).indices # b, t, 1, f, n


    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t


    target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    errors = torch.from_numpy(errors).cuda()
    errors_select = torch.gather(errors, 2, selec_ind) #b, t, 1, f, n
    from einops import rearrange
    errors_select = rearrange(errors_select, 'b t h f n  -> t h b f n', )
    errors_select = errors_select.reshape(t_sz, -1)
    errors_select = torch.mean(errors_select, dim=-1, keepdim=False)
    #errors = errors.transpose(1, 2, 0, 3, 4)  # t, h, b, f, n
    errors_select = errors_select.cpu().numpy()

    return errors_select


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    # velocity_predicted = torch.diff(predicted, dim=axis)
    # velocity_target = torch.diff(target, dim=axis)
    assert axis == 1
    velocity_predicted = predicted[:, 1:,:,:] - predicted[:, :-1,:,:]
    velocity_target = target[:, 1:, :, :] - target[:, :-1, :, :]

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

def mean_velocity_error(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=axis)
    velocity_target = np.diff(target, axis=axis)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))
