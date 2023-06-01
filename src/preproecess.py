from PIL import Image
import numpy as np
import os
from pathlib import Path
from .types import *

'''
[  0,   0,   5],
[255,   0, 255],
[254,   0,   0],
[  0, 255,   0],
[180, 180, 208],
[  0,   0,   0],
[  0, 179, 255],
[254, 255, 189],
[189, 129, 119],
[164, 254, 242]]

'''


RGB2IDX = [
    [0, 0, 0],
    [255, 255, 0],
    [199, 199, 199],
    [106, 106, 106],
    [175, 142,  87],
    [247, 178, 173],
    [164, 255, 242],
    [0, 0, 207],
    [93, 64, 0],
    [255, 138, 0],
    [255, 0, 0],
    [0, 255, 0],
    [255, 0, 255],
    [0, 178, 255],
    [141, 0, 46]
]

class Preprocesser():
    def __init__(self, 
                 paths : List[str],
                 rgb2idx : Dict[int,RGB]= {idx: i for idx, i in enumerate(RGB2IDX)}):
        self.paths = paths
        self.colors_2_cnt : Dict[int, RGB]= {}
        self._rgb2idx : Dict[int,RGB] = rgb2idx

    @property
    def rgb2idx(self):
        return self._rgb2idx

    def rgb2idxByNearest(self, rgb: RGB) -> int:    
        centers_init = np.array(list(self._rgb2idx.values()))
        return np.argmin(((centers_init.astype(np.float32) - rgb)**2).sum(-1)**(0.5))
    
    def makeSemanticSegmentaion(self, images : np.ndarray):
        
        return 
    
    def summaryRGB(self):
        
        '''
        모든 이미지들에서 rgb - cnt mapper 만듬
        '''
        return 
    
    def _summaryCountFromImageFolder(self, 
                                     image_folder_path : Path) -> List[Tuple[RGB, int]]:
        image_folder_path = Path(image_folder_path)
        images = os.listdir(image_folder_path)
        for image_path in images:
            # colors_cnt_pair = 
            
            return list(zip(*np.unique(list(np.array(Image.open(image_folder_path / image_path)).reshape(-1,3)),
                    axis=0,
                    return_counts=True)))
        return colors_cnt_pair
    
    def buildMapper(self):
        '''
        '''
        return 
    