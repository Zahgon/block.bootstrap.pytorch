import os
from os import path as osp
import json
from tqdm import tqdm
import cv2
import torch
import numpy as np
import itertools
from torch.utils.data.sampler import WeightedRandomSampler
from bootstrap.lib.logger import Logger
from bootstrap.datasets import transforms
from bootstrap.datasets.dataset import Dataset

class VRD(Dataset):
    """Documentation for VRD

    """
    def __init__(self,
            dir_data,
            split,
            neg_ratio=0.,
            batch_size=100,
            nb_threads=0,
            seed=1234,
            shuffle=True,
            pin_memory=True,
            mode='predicate'):
        raise NotImplementedError

    def load_json(self):
        pass

    def load_vocabs(self):
        pass

    def extract_vocab(self, path, bg=True):
        pass

    def process_json(self):
        pass

    def process_split(self, split):
        pass

    def remove_no_bboxes_images(self):
        pass

    def download_json(self):
        pass

    def download_features(self):
        pass

    def download_images(self):
        pass

    def make_train_val_split(self, split_ratio=0.95):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def getitem_rel_phrase(self, index):
        pass

    def getitem_predicate(self, index):
        pass

    def __len__(self):
        raise NotImplementedError
