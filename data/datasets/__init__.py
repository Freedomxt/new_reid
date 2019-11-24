from .market1501 import Market1501
from .data_loader import ImageDataset,ImageDatasetSoft
__factory = {
    'market1501':Market1501
}

def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)