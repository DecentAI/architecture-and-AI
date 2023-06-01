from pathlib import Path
import matplotlib.colors as mcolors
from .preproecess import RGB2IDX
import numpy as np
#
from torch_geometric.utils import from_networkx


hex2255 = lambda x : list(map(lambda x : int(x * 255), x))
COLORS_MAP = np.array(list(map(lambda x : hex2255(x),
    list(map(lambda x : mcolors.to_rgb(x),
                list(mcolors.CSS4_COLORS.values())[:len(RGB2IDX)])))))

def MyFilter(array, f_filter):
    return list(filter(f_filter, array))

def getSegmentationPathFromImagePath(image_path : Path) -> Path:
    return str(image_path).replace('BW', 'Segmentation').replace('jpg', 'png')

def showSegmentationMap(segmap, labels):
    color_map = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
    for idx in range(len(labels)):
        color_map[segmap == idx, :] = COLORS_MAP[idx]
    return color_map

def from_networkx_to_torch_graph(networkx_graph):
    
    return 