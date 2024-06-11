# encoding: utf-8
"""
@author:  sherlock (changed by Nir)
@contact: sherlockliao01@gmail.com
"""


import os
import glob
import warnings
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class AICUP(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = ''
    dataset_name = "AICUP"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AICUP-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AICUP-ReID".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.extra_gallery = False

        required_files = [
            self.data_dir,
            self.train_dir,
            # self.query_dir,
            # self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
                          (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])

        super(AICUP, self).__init__(train, query, gallery, **kwargs)

    import os
    import glob

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(os.path.join(dir_path, '*.bmp'))
        data = []
        
        for img_path in img_paths:
            file_name = img_path.split(os.sep)[-1]
            
            # 用下划线拆分文件名
            path_txt = file_name.split('_')
            
            # 对包含 'copy' 和不包含 'copy' 的情况统一处理，直接提取 pid 和 frame_id
            # 对于文件名格式 TrackID_TimeStemp_FrameID_acc_data[_copyN].bmp，TrackID 和 FrameID 分别是第一个和第三个元素

            # 特殊处理来分离拓展名和可能的 copy 后缀
            base_name, _, ext = path_txt[-1].rpartition('.')
            if 'copy' in base_name:
                path_txt[-1] = base_name.split('_copy')[0]
            
            # 尝试提取和转换 pid 和 frame_id
            try:
                pid, frame_id = int(path_txt[0]), int(path_txt[2])
            except ValueError:
                print("error")
                continue  # 如果无法转换为整数，则忽略这张图片

            if pid == -1:
                continue  # 忽略无效图片

            frame_id -= 1  # 索引从0开始
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                frame_id = self.dataset_name + "_" + str(frame_id)

            data.append((img_path, pid, frame_id))
        # print(len(data))
        
        return data
