import os
import re
import sys
import json
import torch
import torch.utils.data as data
import numpy as np
from os import path as osp
from tqdm import tqdm
from collections import Counter
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.datasets.dataset import Dataset
from bootstrap.datasets import transforms as bootstrap_tf
from bootstrap.datasets.dataset import ListDatasets

def tokenize(sentence):
    pass

def tokenize_mcb(s):
    pass


class AbstractVQA(Dataset):

    def __init__(self,
            dir_data='/local/cadene/data/vqa',
            split='train', 
            batch_size=80,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            has_valset=True,
            has_testset=True,
            has_testset_anno=False,
            has_testdevset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False):
        raise NotImplementedError

    def add_word_tokens(self, word_to_wid):
        pass

    def bootstrapping(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def get_image_name(self, image_id='1', format='COCO_%s_%012d.jpg'):
        pass

    def name_subdir_processed(self):
        pass

    def get_subdir_processed(self):
        pass

    def get_subtype(self, testdev=False):
        pass

    ################################################################################
    # Preprocessing

    def download(self):
        raise NotImplementedError()

    def process(self):
        pass

    def tokenize_answers(self, annotations):
        pass

    def add_image_names(self, dataset):
        pass

    def add_answer(self, annotations):
        pass

    def top_answers(self, annotations, nans):
        pass

    def annotations_in_top_answers(self, annotations, questions, top_answers):
        pass

    def tokenize_questions(self, questions, nlp):
        pass

    def top_words(self, questions, minwcount):
        pass

    def merge_annotations_with_questions(self, ann, ques):
        pass

    def insert_UNK_token(self, questions, wcounts, minwcount):
        pass

    def encode_questions(self, questions, word_to_wid):
        pass

    def encode_answers(self, annotations, ans_to_aid):
        pass

    def add_answers_occurence(self, annotations, ans_to_aid):
        # for samplingans during training
        pass

    #############################################################################""
    # Preprocessing on a list dataset (vqa2+vgenome setup)

    def sync_from(self, dataset):
        pass


class ListVQADatasets(ListDatasets):

    def __init__(self,
             datasets,
             split='train',
             batch_size=4,
             shuffle=False,
             pin_memory=False,
             nb_threads=4,
             seed=1337):
        raise NotImplementedError
        

    def make_subdir_processed(self):
        pass

    def process(self):
        pass

    def get_subtype(self):
        pass

