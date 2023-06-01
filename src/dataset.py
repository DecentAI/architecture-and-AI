import torch
import numpy as np
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from .types import *
from copy import deepcopy
from functools import reduce 

class WallSectionDataset(Dataset):
    def __init__(self, 
                root_paths : List[str],
                original_image_folder_name : str = 'BW',
                segmentation_map_folder_name : str = 'Segmentation',
                transform = None):
        self._root_paths = [Path(_path) for _path in root_paths]
        self._segmentation_map_folder_name = segmentation_map_folder_name
        self._originl_image_path = [[each_path / original_image_folder_name / image_path for image_path in os.listdir(each_path / original_image_folder_name) if image_path != '..DS_Store']
                                    for each_path in self._root_paths]
        self._originl_image_path = [i for i in reduce(lambda x,y : x+y, self._originl_image_path)]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self._originl_image_path)
    
    def __getitem__(self, idx : int):
        image_path = self._originl_image_path[idx]
        image = np.array(Image.open(image_path))
        seg_path = list(image_path.parts)
        seg_path[-2] = self._segmentation_map_folder_name
        seg_path = Path('/'.join(seg_path).replace('.jpg', '.png'))
        seg = np.array(Image.open(seg_path))
        if self.transform:
            agumented = self.transform(image=image, mask=seg)
            image, seg = agumented['image'], agumented['mask']
        image_CWH = np.transpose(image, axes=[2,0,1]) 
        return image_CWH, seg
    

def splitDataset(wallSectionDataset : WallSectionDataset, rv_seed : int = 224, train_ratio : float = 0.8) -> Tuple[WallSectionDataset, WallSectionDataset]:
    np.random.seed(rv_seed)
    WALL_TYPE_LIST = ['WD', 'tile', 'cStud', 'cmu']
    train_data_paths, test_data_paths = [], []
    
    for wall_type in WALL_TYPE_LIST:
        wall_type_paths = [i for i in wallSectionDataset._originl_image_path if wall_type in i.parts]
        np.random.shuffle(wall_type_paths)
        train_data_paths.extend(wall_type_paths[:int(len(wall_type_paths)*train_ratio)])
        test_data_paths.extend(wall_type_paths[int(len(wall_type_paths)*train_ratio):])
    train_wallSectionDataset = deepcopy(wallSectionDataset)
    train_wallSectionDataset._originl_image_path = train_data_paths
    test_wallSectionDataset = deepcopy(wallSectionDataset)
    test_wallSectionDataset._originl_image_path = test_data_paths 
    return train_wallSectionDataset, test_wallSectionDataset