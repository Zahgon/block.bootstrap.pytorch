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
        super(AbstractVQA, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle)
        self.nans = nans
        self.minwcount = minwcount
        self.nlp = nlp
        self.proc_split = proc_split
        self.samplingans = samplingans
        # preprocessing
        self.has_valset = has_valset
        self.has_testset = has_testset
        self.has_testset_anno = has_testset_anno
        self.has_testdevset = has_testdevset
        self.has_answers_occurence = has_answers_occurence
        self.do_tokenize_answers = do_tokenize_answers

        # sanity checks
        if self.split in ['test', 'val'] and self.samplingans:
            raise ValueError()

        self.dir_raw = os.path.join(self.dir_data, 'raw')
        if not os.path.exists(self.dir_raw):
            self.download()

        self.dir_processed = os.path.join(self.dir_data, 'processed')
        self.subdir_processed = self.get_subdir_processed()
        self.path_wid_to_word = osp.join(self.subdir_processed, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subdir_processed, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subdir_processed, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subdir_processed, 'ans_to_aid.pth')
        self.path_trainset = osp.join(self.subdir_processed, 'trainset.pth')
        self.path_valset = osp.join(self.subdir_processed, 'valset.pth')
        self.path_is_qid_testdev = osp.join(self.subdir_processed, 'is_qid_testdev.pth')
        self.path_testset = osp.join(self.subdir_processed, 'testset.pth')
        
        if not os.path.exists(self.subdir_processed):
            self.process()

        self.wid_to_word = torch.load(self.path_wid_to_word)
        self.word_to_wid = torch.load(self.path_word_to_wid)
        self.aid_to_ans = torch.load(self.path_aid_to_ans)
        self.ans_to_aid = torch.load(self.path_ans_to_aid)

        if 'train' in self.split:
            self.dataset = torch.load(self.path_trainset)
        elif self.split == 'val':
            if self.proc_split == 'train':
                self.dataset = torch.load(self.path_valset)
            elif self.proc_split == 'trainval':
                self.dataset = torch.load(self.path_trainset)
        elif self.split == 'test':
            self.dataset = torch.load(self.path_testset)
            if self.has_testdevset:
                self.is_qid_testdev = torch.load(self.path_is_qid_testdev)

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.PadTensors(use_keys=[
                'question', 'pooled_feat', 'cls_scores', 'rois', 'cls', 'cls_oh', 'norm_rois'
            ]),
            #bootstrap_tf.SortByKey(key='lengths'), # no need for the current implementation
            bootstrap_tf.StackTensors()
        ])

        if self.proc_split == 'trainval' and self.split in ['train','val']:
            self.bootstrapping()

    def add_word_tokens(self, word_to_wid):
        pass

    def bootstrapping(self):
        pass

    def __len__(self):
        return len(self.dataset['questions'])

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
        super(ListVQADatasets, self).__init__(
            datasets=datasets,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            bootstrapping=False,
            seed=seed)

        self.subdir_processed = self.make_subdir_processed()
        Logger()('Subdir proccessed: {}'.format(self.subdir_processed))
        self.path_wid_to_word = osp.join(self.subdir_processed, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subdir_processed, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subdir_processed, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subdir_processed, 'ans_to_aid.pth')

        self.process()
        
        # if not os.path.isdir(self.subdir_processed):
        #     self.process()
        # else:
        #     Logger()('Loading list_datasets_vqa proccessed state')
        #     self.wid_to_word = torch.load(self.path_wid_to_word)
        #     self.word_to_wid = torch.load(self.path_word_to_wid)
        #     self.aid_to_ans = torch.load(self.path_aid_to_ans)
        #     self.ans_to_aid = torch.load(self.path_ans_to_aid)

        #     for i in range(len(self.datasets)):
        #         subdir_processed = os.path.join(self.subdir_processed, '{}.{}'.format(
        #             self.datasets[i].__class__.__name__,
        #             self.datasets[i].split))
        #         path_dataset = os.path.join(subdir_processed, 'dataset.pth')
        #         self.datasets[i].dataset = torch.load(path_dataset)
        #     Logger()('Done !')

        Logger()('Final number of tokens {}'.format(len(self.word_to_wid)))

        self.make_lengths_and_ids()
        

    def make_subdir_processed(self):
        pass

    def process(self):
        pass

    def get_subtype(self):
        pass

