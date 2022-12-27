import os
import re
import cv2 as cv
import numpy as np
from dataset.base import StereoPairBase


class StereoPairMiddlebury(StereoPairBase):

    def read(self):
        self.read_pair_images()
        self.read_calib_info()
        self.read_disparity_map()

    def read_pair_images(self):
        self.src_img = cv.imread(os.path.join(self.root_path, 'im0.png'),
                                 flags=cv.IMREAD_COLOR)
        self.dst_img = cv.imread(os.path.join(self.root_path, 'im1.png'),
                                 flags=cv.IMREAD_COLOR)

    def read_calib_info(self):
        calib_file = os.path.join(self.root_path, 'calib.txt')

        with open(calib_file, 'r') as fp:
            for line in fp.readlines():
                header, params = line.strip().split('=')

                if header == 'cam0':
                    self.src_cam = np.array(
                        [[float(x) for x in row.split()]
                         for row in params.lstrip('[').rstrip(']').split(';')],
                        dtype=np.float32)
                elif header == 'cam1':
                    self.dst_cam = np.array(
                        [[float(x) for x in row.split()]
                         for row in params.lstrip('[').rstrip(']').split(';')],
                        dtype=np.float32)
                else:
                    self.__setattr__(header, float(params))

    def read_disparity_map(self):
        self.src_disp, self.src_scale = StereoPairMiddlebury.read_pfm_file(
            os.path.join(self.root_path, 'disp0.pfm'))
        self.dst_disp, self.dst_scale = StereoPairMiddlebury.read_pfm_file(
            os.path.join(self.root_path, 'disp1.pfm'))

    @staticmethod
    def read_pfm_file(pfm_file):
        with open(pfm_file, 'rb') as fp:
            header = fp.readline().decode().rstrip()
            c = 3 if header == 'PF' else 1

            dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                                 fp.readline().decode('utf-8'))
            if dim_match:
                w, h = map(int, dim_match.groups())
            else:
                raise Exception('malformed PFM format!')

            scale = float(fp.readline().decode().rstrip())
            if scale < 0:
                endian = '<'  # little endian
                scale = -scale
            else:
                endian = '>'  # big endian

            disp = np.fromfile(fp, endian + 'f')

        disp = np.reshape(disp, newshape=[h, w, c])
        disp = np.flipud(disp).astype('uint8')
        return disp, scale