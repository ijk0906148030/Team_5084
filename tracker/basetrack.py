import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0       # 新追踪对象
    Tracked = 1    # 已确认并正在追踪的对象
    Lost = 2       # 暂时失去追踪的对象
    Tentative = 3  # 疑似追踪对象（新增状态）
    LongLost = 4   # 长时间失去追踪的对象
    Removed = 5    # 移除的追踪对象


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0
