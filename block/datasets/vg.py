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
from .vqa_utils import AbstractVQA

class VG(AbstractVQA):

    def __init__(self,
            dir_data='data/vg',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            dataset_img=None,
            nans=2000,
            minwcount=10,
            nlp='mcb',
            dir_rcnn='data/vqa/vgenome/extract_rcnn'):
        raise NotImplementedError

    def add_rcnn_to_item(self, item):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        pass

    def make_annotations(self):
        # transform vgenome annotations into vqa2 format
        pass
