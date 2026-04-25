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

class TDIUC(AbstractVQA):

    def __init__(self,
            dir_data='data/tdiuc',
            split='train',
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            dir_rcnn='data/tdiuc/extract_rcnn'):
        raise NotImplementedError

    def add_answer(self, annotations):
        pass

    def add_rcnn_to_item(self, item):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        pass
