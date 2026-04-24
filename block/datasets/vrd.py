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
        super(VRD, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle)
        assert(split in ['train', 'val', 'test', 'trainval'])
        self.neg_ratio = neg_ratio
        self.seed = seed
        assert(mode in ['predicate', 'rel_phrase'])
        self.mode = mode
        self.dir_raw_json = osp.join(self.dir_data, 'annotations','raw')
        self.dir_images = osp.join(self.dir_data, 'images')
        self.dir_processed = osp.join(self.dir_data, 'annotations','processed')

        if not osp.exists(self.dir_raw_json):
            self.download_json()
        if not osp.exists(self.dir_images):
            self.download_images()

        self.vocabs = self.load_vocabs()
        if not osp.exists(self.dir_processed):
            self.process_json()

        self.json = self.load_json()
        self.ids = sorted(list(self.json.keys()))
        self.ids = self.remove_no_bboxes_images()

        if self.mode == 'predicate':
            if self.split in ['train', 'val']:
                self.make_train_val_split()
            if self.split in ['train', 'val', 'trainval']:
                self.dir_features = osp.join(self.dir_data, 'features', 'gt_boxes', 'train')
            else:
                self.dir_features = osp.join(self.dir_data, 'features', 'gt_boxes', 'test')

        elif self.mode == 'rel_phrase':
            assert(self.split == 'test')
            self.dir_features = osp.join(self.dir_data, 'features', 'pred_boxes', 'test')
            path_jraw = osp.join(self.dir_raw_json, 'annotations_test.json')
            with open(path_jraw, 'r') as f:
                self.json_raw = json.load(f)

        if not osp.exists(self.dir_features):
            self.download_features()

        if self.split in ['train', 'trainval']:
            self.shuffle = False
            self.sampler = WeightedRandomSampler(
                weights=[1]*len(self),
                num_samples=len(self),
                replacement=True)
        else:
            self.sampler = None

        self.collate_fn = transforms.Compose([
            transforms.ListDictsToDictLists(),
            transforms.CatTensors()
        ])

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
        if self.mode == 'predicate':
            item = self.getitem_predicate(index)
        elif self.mode == 'rel_phrase':
            item = self.getitem_rel_phrase(index)
        else:
            raise ValueError(self.mode)
        return item

    def getitem_rel_phrase(self, index):
        pass

    def getitem_predicate(self, index):
        pass

    def __len__(self):
        return len(self.ids)
