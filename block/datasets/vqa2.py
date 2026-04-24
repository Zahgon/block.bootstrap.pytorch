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
        super(VQA2, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split=proc_split,
            samplingans=samplingans,
            has_valset=True,
            has_testset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False)
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
        item['image_name'] = question['image_name']

        # Process Object, Attribut and Relational features
        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            if self.load_original_annotation:
                item['original_annotation'] = annotation
            if 'train' in self.split and self.samplingans:
                proba = annotation['answers_count']
                proba = proba / np.sum(proba)
                item['answer_id'] = int(np.random.choice(annotation['answers_id'], p=proba))
            else:
                item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']
        else:
            if item['question_id'] in self.is_qid_testdev:
                item['is_testdev'] = True
            else:
                item['is_testdev'] = False
        return item

    def download(self):
        pass
