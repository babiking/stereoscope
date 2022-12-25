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


def intersect_two_lines_2d(k_0, b_0, k_1, b_1):
    # line equation: kx + b - y = 0
    # y = k_0 x + b_0
    # y = k_1 x + b_1

    # (k_0 - k_1) x + (b_0 - b_1) = 0
    if abs(k_0 - k_1) < 1e-3:
        return

    x = -(b_0 - b_1) / (k_0 - k_1)
    y = k_0 * x + b_0
    return np.array([x, y], dtype=np.float32)


def get_line_by_points(p_0, p_1):
    x_0, y_0 = p_0
    x_1, y_1 = p_1

    k = (y_1 - y_0) / (x_1 - x_0)
    b = y_0 - k * x_0
    # kx - y + b = 0
    return (k, -1, b)


def get_perpendicular_foot(a, b, c, p):
    # ax + by + c = 0
    m, n = p

    x = (b * b * m - a * b * n - a * c) / (a * a + b * b)
    y = (a * a * n - a * b * m - b * c) / (a * a + b * b)
    return np.array([x, y], dtype=np.float32)


def intersect_two_rays_3d(o_0, dir_0, o_1, dir_1):
    w_0 = (o_0 - o_1)

    a = np.dot(dir_0, dir_0)
    b = np.dot(dir_0, dir_1)
    c = np.dot(dir_1, dir_1)
    d = np.dot(dir_0, w_0)
    e = np.dot(dir_1, w_0)

    s_0 = (b * e - c * d) / (a * c - b**2 + 1e-12)
    s_1 = (a * e - b * d) / (a * c - b**2 + 1e-12)

    p_0 = o_0 + s_0 * dir_0
    p_1 = o_1 + s_1 * dir_1

    dist = np.linalg.norm(p_0 - p_1)
    return p_0, p_1, dist


def get_vanish_point(p_0, p_1, dim_u, dim_v):
    cp = np.array([dim_u / 2, dim_v / 2], dtype=np.float32)

    k_0, b_0 = fit_linear_scatter_points(p_0[:, 0], p_0[:, 1])
    k_1, b_1 = fit_linear_scatter_points(p_1[:, 0], p_1[:, 1])
    vp = intersect_two_lines_2d(k_0, b_0, k_1, b_1)

    sign = 1.0
    if vp is not None:
        ray_c = vp - cp
        ray_c /= np.linalg.norm(ray_c)
        ray_p = p_0[-1, :] - p_0[0, :]
        ray_p /= np.linalg.norm(ray_p)
        sign = 1.0 if np.dot(ray_c, ray_p) >= 0.0 else -1.0
    return vp, sign, k_0, b_0, k_1, b_1


def calibrate_by_two_vanish_points(
    vp_0,
    vp_sign_0,
    vp_1,
    vp_sign_1,
    w_obj,
    w_pix,
    s_obj,
    s_pix,
    dim_u,
    dim_v,
):
    cp = np.array([dim_u / 2.0, dim_v / 2.0], dtype=np.float32)

    # 1. calculate distance between camera origin and image plane
    vp_i = get_perpendicular_foot(*get_line_by_points(vp_0, vp_1), cp)

    cp_to_vp_i = np.linalg.norm(vp_i - cp)
    vp_0_to_i = np.linalg.norm(vp_i - vp_0)
    vp_1_to_i = np.linalg.norm(vp_i - vp_1)

    f = np.sqrt(vp_0_to_i * vp_1_to_i - cp_to_vp_i**2)

    # 2. reproject vanish points from image plane to camera coordinate
    # u = au * fx * x + cx
    # v = av * fx * y + cy
    vp_y = np.concatenate([vp_0 - cp, [f]], dtype=np.float32)
    vp_y /= vp_sign_0 * np.linalg.norm(vp_y)
    vp_x = np.concatenate([vp_1 - cp, [f]], dtype=np.float32)
    vp_x /= vp_sign_1 * np.linalg.norm(vp_x)
    vp_z = np.cross(vp_x, vp_y)
    cam_rot_mat = R.from_matrix(np.vstack([vp_x, vp_y, vp_z]).T)
    cam_rot_quat = Quaternion(matrix=cam_rot_mat.as_matrix())

    # 3. use 2 known object points to estimate the scale
    cam_w_s_dist = np.linalg.norm(s_obj - w_obj)
    cam_w_s_dir = cam_rot_quat.rotate(s_obj - w_obj)
    cam_w_s_dir /= np.linalg.norm(cam_w_s_dir)
    cam_w_pnt = np.concatenate([w_pix - cp, [f]], dtype=np.float32)
    cam_s_pnt = np.concatenate([s_pix - cp, [f]], dtype=np.float32)
    cam_s_dir = cam_s_pnt / np.linalg.norm(cam_s_pnt)

    # a ray starting from cam_w_pnt, along the direction of cam_w_s_dir
    # another ray starting from camera origin, along the direction of cam_s_dir
    # find out least-square-error intersection point of 2 rays
    cam_q_pnt, _, ray_dist = intersect_two_rays_3d(
        cam_w_pnt,
        cam_w_s_dir,
        [0.0, 0.0, 0.0],
        cam_s_dir,
    )

    cam_w_dist = np.linalg.norm(cam_w_pnt) * cam_w_s_dist / np.linalg.norm(
        cam_q_pnt - cam_w_pnt)
    cam_t_vec = cam_w_dist * cam_w_pnt / (np.linalg.norm(cam_w_pnt) + 1e-12)
    return f, cam_rot_quat, cam_t_vec


def main():
    # 1. setup chessboard object, i.e. surface point coordinates
    cb_dim_h = 13
    cb_dim_w = 17
    cb_sp_h = 1.0
    cb_sp_w = 1.0
    cb_pnts = get_chessboard_points_3d(cb_dim_h, cb_dim_w, cb_sp_h, cb_sp_w)

    # 2. setup camera intrinsic parameters
    au = 1.0  # unit: pixel / m
    av = 1.0
    f = 312.0
    shear = 0.0
    cam_dim_w = 1080
    cam_dim_h = 720
    cam_mat = setup_camera_intrinsics(au, av, f, shear, cam_dim_w, cam_dim_h)

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
        cb_pixs += np.random.randn(*cb_pixs.shape) * 0.3

        # 5. linear fit chessboard 4 edges and find out 2 valid vanish points
        u_vp, u_vp_sign, u_k_0, u_b_0, u_k_1, u_b_1 = get_vanish_point(
            p_0=cb_pixs[:, 0, :],
            p_1=cb_pixs[:, -1, :],
            dim_u=cam_dim_h,
            dim_v=cam_dim_w)
        v_vp, v_vp_sign, v_k_0, v_b_0, v_k_1, v_b_1 = get_vanish_point(
            p_0=cb_pixs[0, :, :],
            p_1=cb_pixs[-1, :, :],
            dim_u=cam_dim_h,
            dim_v=cam_dim_w)
        if u_vp is None or v_vp is None:
            continue

        # 6. get 2 object points:
        #   a. world coordinate system center
        #   b. randomly selected object point
        w_obj = [0.0, 0.0, 0.0]
        w_pix = cb_pixs[(cb_dim_h - 1) // 2, (cb_dim_w - 1) // 2, :]
        i = 0
        j = 0
        while (i == 0 and j == 0):
            i = np.random.randint(low=0, high=cb_dim_h)
            j = np.random.randint(low=0, high=cb_dim_w)
        s_obj = cb_pnts[i, j, :]
        s_pix = cb_pixs[i, j, :]

        calib_f, calib_rot_quat, calib_t_vec = calibrate_by_two_vanish_points(
            u_vp, u_vp_sign, v_vp, v_vp_sign, w_obj, w_pix, s_obj, s_pix,
            cam_dim_h, cam_dim_w)

        f_err = abs(calib_f - f)
        # assert f_err < 1e-3, 'focal length calibration error!'

        rot_quat_err = (cam_rot_quat.inverse * calib_rot_quat).rotation_matrix
        # assert np.allclose(rot_quat_err,
        #                    np.eye(3, dtype=np.float32),
        #                    rtol=1e-5,
        #                    atol=1e-5), 'rotation matrix calibration error!'

        t_vec_err = np.linalg.norm(cam_t_vec - calib_t_vec)
        # assert t_vec_err < 1e-4, 'translation vector calibration error!'

        draw = True
        if draw:
            plt.figure()
            plt.scatter(cb_pixs[:, :, 0].flatten(),
                        cb_pixs[:, :, 1].flatten(),
                        color='blue',
                        marker='*')
            xs = np.arange(0.0, cam_dim_h, step=0.05)
            plt.plot(xs, u_k_0 * xs + u_b_0, color='red', label='u-axis:0')
            plt.plot(xs, u_k_1 * xs + u_b_1, color='black', label='u-axis:1')
            plt.plot(xs, v_k_0 * xs + v_b_0, color='green', label='v-axis:0')
            plt.plot(xs, v_k_1 * xs + v_b_1, color='yellow', label='v-axis:1')
            plt.xlim(0, cam_dim_h)
            plt.ylim(0, cam_dim_w)
            plt.legend(loc='best')
            roundn = lambda array, n: [round(float(x), n) for x in array]
            plt.title(
                f'focal length: {f} VS {calib_f:.4f}' + '\n' \
                    + f'rotation: {roundn(cam_rot_quat.elements, 2)} VS {roundn(calib_rot_quat, 2)}' + '\n' \
                        +f'translation: {roundn(cam_t_vec, 3)} VS {roundn(calib_t_vec, 3)}.'
            )
            plt.show()


if __name__ == '__main__':
    main()
