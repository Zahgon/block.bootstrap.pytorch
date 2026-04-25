import os
import os.path as osp
import sys
import csv
import base64
import json
import numpy as np
import torch
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.datasets.dataset import Dataset
from .vqa2 import VQA2
from .vg import VG

class VQA2VG(Dataset):

    def __init__(self,
            dir_data='data/vqa2',
            dir_data_vg='data/vg',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            dir_rcnn='data/coco/extract_rcnn',
            dir_rcnn_vg='data/vg/extract_rcnn'):
        raise NotImplementedError

    def __getattr__(self, key):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
