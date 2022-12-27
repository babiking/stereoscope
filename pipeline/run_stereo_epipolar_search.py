from dataset.middlebury import StereoPairMiddlebury
from match.image_feature_match import match_keypoints_2d
from convert.ply_format import write_vertices_to_ply


def main():
    bury = StereoPairMiddlebury(root_path='data/bandsaw1')

    # src_kp, src_desc, \
    #     dst_kp, dst_desc, good = match_keypoints_2d(bury.src_img, bury.dst_img, 0.75, False)

    # for match in good:
    #     i1 = match[0].queryIdx
    #     i2 = match[0].trainIdx

    #     w1, h1 = src_kp[i1].pt
    #     w2, h2 = dst_kp[i2].pt

    #     disp1 = bury.src_disp[int(h1), int(w1)]
    #     disp2 = bury.dst_disp[int(h2), int(w2)]

    vertices = []
    for u in range(int(bury.height)):
        for v in range(int(bury.width)):
            b, g, r = bury.src_img[u, v]

            disp = bury.src_disp[u, v]

            if abs(disp) < 1e-6:
                continue

            z = (bury.baseline * bury.src_cam[0, 0]) / (disp + bury.doffs)
            # u = x / z * fx + cx
            # v = y / z * fy + cy
            # [x, y, z] -> [x', y', 1]
            x = (u - bury.src_cam[0, 2]) / bury.src_cam[0, 0] * z
            y = (v - bury.src_cam[1, 2]) / bury.src_cam[1, 1] * z

            vertices.append(tuple([x, y, z, r, g, b]))

    write_vertices_to_ply(vertices, 'sample.ply')


if __name__ == '__main__':
    main()