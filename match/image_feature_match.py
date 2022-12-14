import cv2 as cv
from matplotlib import pyplot as plt


def match_keypoints_2d(src_img, dst_img, threshold=0.75, draw=False):
    src_gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    dst_gray = cv.cvtColor(dst_img, cv.COLOR_BGR2GRAY)

    # 1. feature extractor / detector
    detector = cv.ORB_create()

    src_kp, src_desc = detector.detectAndCompute(src_gray, None)
    dst_kp, dst_desc = detector.detectAndCompute(dst_gray, None)

    # 2. feature descriptor match
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(src_desc, dst_desc, k=2)

    # 3. filter good matches
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    if draw:
        cat_img = cv.drawMatchesKnn(
            src_img,
            src_kp,
            dst_img,
            dst_kp,
            good,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(cat_img[:, :, ::-1])
        plt.show()
    return src_kp, src_desc, dst_kp, dst_desc, good
