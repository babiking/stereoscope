class StereoPairBase(object):

    def __init__(self, root_path):
        self.root_path = root_path
        self.read()

    def read(self):
        raise NotImplementedError

    def read_pair_images(self):
        raise NotImplementedError

    def read_calib_info(self):
        raise NotImplementedError

    def read_disparity_map(self):
        raise NotImplementedError

    def read_depth_map(self):
        raise NotImplementedError

    def read_object_points(self):
        raise NotImplementedError