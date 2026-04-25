import os
import csv
import copy
import json
import torch
import numpy as np
from os import path as osp
from bootstrap.lib.logger import Logger
from .vqa_utils import AbstractVQA

class VQA2(AbstractVQA):

    def __init__(self,
            dir_data='data/vqa2',
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
            dir_rcnn='data/coco/extract_rcnn'):
        raise NotImplementedError

    def add_rcnn_to_item(self, item):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        pass
