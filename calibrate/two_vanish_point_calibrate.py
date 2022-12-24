import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def get_chessboard_points_3d(cb_dim_h=13,
                             cb_dim_w=17,
                             cb_sp_h=1.0,
                             cb_sp_w=1.0):
    cb_ys, cb_xs = np.meshgrid(range(-(cb_dim_h - 1) // 2,
                                     (cb_dim_h - 1) // 2 + 1),
                               range(-(cb_dim_w - 1) // 2,
                                     (cb_dim_w - 1) // 2 + 1),
                               indexing='ij')

    cb_pnts = np.zeros(shape=[cb_dim_h, cb_dim_w, 3], dtype=np.float32)
    cb_pnts[:, :, 0] = cb_xs * cb_sp_w
    cb_pnts[:, :, 1] = cb_ys * cb_sp_h
    return cb_pnts


def setup_camera_intrinsics(au=1.0,
                            av=1.0,
                            f=312.0,
                            shear=0.0,
                            cam_dim_w=1080,
                            cam_dim_h=720):
    cam_mat = np.array([
        [au * f, shear, cam_dim_h / 2.0],
        [0.0, av * f, cam_dim_w / 2.0],
        [0.0, 0.0, 1.0],
    ],
                       dtype=np.float32)
    return cam_mat


def get_direction_from_angles(polar, azimuth):
    nz = np.cos(polar)
    nx = np.sin(polar) * np.cos(azimuth)
    ny = np.sin(polar) * np.sin(azimuth)

    dir = np.array([nx, ny, nz], dtype=np.float32)
    dir /= (np.linalg.norm(dir) + 1e-12)
    return dir


def span_coordinate_axis_3d(axis):
    e_i = axis / np.linalg.norm(axis)
    e_j = np.array([-e_i[1], e_i[0], 0.0]) / np.linalg.norm(e_i[:2])
    e_k = np.cross(e_i, e_j)
    # make sure rotation matrix is orthogonal with tolerance < 1e-8
    rotor = R.from_matrix(np.vstack([e_i, e_j, e_k]).T)
    return rotor.as_matrix()


def setup_camera_extrinsics(cam_t_vec, cam_polar, cam_azimuth):
    cam_z_axis = -get_direction_from_angles(cam_polar, cam_azimuth)

    cam_rot_mat = span_coordinate_axis_3d(cam_z_axis)
    # axis order ZXY -> XYZ
    cam_rot_mat = cam_rot_mat[:, [1, 2, 0]]

    cam_rot_quat = Quaternion(matrix=cam_rot_mat)
    return cam_rot_quat, cam_t_vec


def interpolate_camera_poses(cam_rot_quat_0,
                             cam_t_vec_0,
                             cam_rot_quat_1,
                             cam_t_vec_1,
                             ratio=0.0):
    cam_rot_quat = Quaternion.slerp(cam_rot_quat_0, cam_rot_quat_1, ratio)

    cam_t_vec = (1.0 - ratio) * cam_t_vec_0 + ratio * cam_t_vec_1
    return cam_rot_quat, cam_t_vec


def inverse_camera_pose(cam_rot_quat, cam_t_vec):
    inv_cam_rot_quat = cam_rot_quat.inverse
    inv_cam_t_vec = -inv_cam_rot_quat.rotate(cam_t_vec)
    return inv_cam_rot_quat, inv_cam_t_vec


def capture_with_camera(w_pnts, cam_rot_quat, cam_t_vec, cam_mat):
    dim_h, dim_w, _ = w_pnts.shape

    c_pnts = \
        cam_rot_quat.rotation_matrix @ w_pnts.reshape([-1, 3]).T + cam_t_vec[:, np.newaxis]
    c_pnts /= c_pnts[-1, :]

    c_pixs = (cam_mat @ c_pnts)
    c_pixs = c_pixs[:2, :].T.reshape([dim_h, dim_w, 2])
    return c_pixs


def fit_linear_scatter_points(xs, ys):
    n = len(xs)
    x_sum = np.sum(xs)
    x2_sum = np.sum(np.power(xs, 2))
    y_sum = np.sum(ys)
    xy_sum = np.sum(xs * ys)

    k = (y_sum * x_sum - n * xy_sum) / (x_sum**2 - n * x2_sum)
    b = (y_sum - k * x_sum) / n
    return k, b


def intersect_linear_lines(k_0, b_0, k_1, b_1):
    # line equation: kx + b - y = 0
    # y = k_0 x + b_0
    # y = k_1 x + b_1

    # (k_0 - k_1) x + (b_0 - b_1) = 0
    if abs(k_0 - k_1) < 1e-3:
        return

    x = -(b_0 - b_1) / (k_0 - k_1)
    y = k_0 * x + b_0
    return np.array([x, y], dtype=np.float32)


def main():
    work_path = os.path.dirname(__file__)

    # 1. setup chessboard object, i.e. surface point coordinates
    cb_pnts = get_chessboard_points_3d()

    # 2. setup camera intrinsic parameters
    cam_mat = setup_camera_intrinsics()

    # 3. setup interpolated camera extrinsic parameters
    # 4. project chessboard points (world) onto camera image plane
    n_cams = 21
    cam_rot_quat_0, cam_t_vec_0 = setup_camera_extrinsics(
        cam_t_vec=np.array([18.0, 8.0, 40.0], dtype=np.float32),
        cam_polar=np.pi / 5,
        cam_azimuth=np.arctan2(4, 3))
    cam_rot_quat_1, cam_t_vec_1 = setup_camera_extrinsics(
        cam_t_vec=np.array([-15.0, -12.0, 40.0], dtype=np.float32),
        cam_polar=np.pi / 6,
        cam_azimuth=np.arctan2(-4, -3))

    for i in tqdm(range(n_cams),
                  desc=f'capture chessboard at each camera pose...'):
        ratio = float(i) / (n_cams - 1)

        inv_cam_rot_quat, inv_cam_t_vec = interpolate_camera_poses(
            cam_rot_quat_0, cam_t_vec_0, cam_rot_quat_1, cam_t_vec_1, ratio)

        cam_rot_quat, cam_t_vec = inverse_camera_pose(inv_cam_rot_quat,
                                                      inv_cam_t_vec)

        cb_pixs = capture_with_camera(
            cb_pnts,
            cam_rot_quat,
            cam_t_vec,
            cam_mat,
        )

        # 5. linear fit chessboard 4 edges and find out 2 valid vanish points
        u_k_0, u_b_0 = fit_linear_scatter_points(cb_pixs[:, 0, 0],
                                                 cb_pixs[:, 0, 1])
        u_k_1, u_b_1 = fit_linear_scatter_points(cb_pixs[:, -1, 0],
                                                 cb_pixs[:, -1, 1])
        u_vp = intersect_linear_lines(u_k_0, u_b_0, u_k_1, u_b_1)

        v_k_0, v_b_0 = fit_linear_scatter_points(cb_pixs[0, :, 0],
                                                 cb_pixs[0, :, 1])
        v_k_1, v_b_1 = fit_linear_scatter_points(cb_pixs[-1, :, 0],
                                                 cb_pixs[-1, :, 1])
        v_vp = intersect_linear_lines(v_k_0, v_b_0, v_k_1, v_b_1)

        if u_vp is None or v_vp is None:
            continue


if __name__ == '__main__':
    main()