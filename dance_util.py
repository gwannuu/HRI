import pickle
from enum import Enum

class Axis(Enum):
    x = 0
    y = 2
    z = 1

def open_files(path):
    with open(f'{path}', 'rb')as f:
        data = pickle.load(f)
    data = data["keypoints3d_optim"]
    print(data.shape)
    data = data[:, :, [Axis.x.value, Axis.y.value, Axis.z.value]]
    print(data.shape)
    return data