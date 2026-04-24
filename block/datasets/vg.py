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
        self.dir_data = dir_data
        self.dir_raw = os.path.join(self.dir_data, 'raw')
        dir_anno = os.path.join(self.dir_raw, 'annotations')
        if not os.path.isdir(dir_anno):
            self.make_annotations()
        super(VG, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split='train',
            samplingans=False,
            has_valset=False,
            has_testset=False,
            has_testset_anno=False,
            has_testdevset=False,
            has_answers_occurence=False,
            do_tokenize_answers=True)
        self.dir_rcnn = dir_rcnn
        # to activate manually in visualization context (notebook)
        self.load_original_annotation = False

    def add_rcnn_to_item(self, item):
        pass

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset['questions'][index]
        if self.load_original_annotation:
            item['original_question'] = question

        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        item['image_name'] = question['image_id']

        # Process Object, Attribut and Relational features
        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            if self.load_original_annotation:
                item['original_annotation'] = annotation
            
            item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']
        return item

    def download(self):
        pass

    def make_annotations(self):
        # transform vgenome annotations into vqa2 format
        pass
