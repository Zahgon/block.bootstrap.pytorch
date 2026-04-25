import os
from os import path as osp
import json
import torch
import torch.utils.data as data

from PIL import Image
import numpy as np
import random

import xml.etree.ElementTree as ET
import cv2
import sys
sys.path.insert(0,'./')
from supervqa.datasets import process_utils as p_utils

from tqdm import tqdm

raw_dir = "/local/benyounes/data/vrd/"
xml_folder = osp.join(raw_dir, 'xml')
split = "train"
img_relative = osp.join('sg_dataset', 'sg_%s_images' % split)
objs_vocab = json.load(open(osp.join(raw_dir,'objects.json')))
preds_vocab = json.load(open(osp.join(raw_dir,'predicates.json')))

def _create_xml(fname, rels):
    # Create root
    raise NotImplementedError

def _convert_vocabs():
    raise NotImplementedError
def main():
    raise NotImplementedError


if __name__ =="__main__":
    main()
    #name, rels, new_tree = main()
    #ex_tree = ET.parse("/local/cadene/data/faster-rcnn.pytorch/vgenome/xml/9.xml")
    #
    #for root in [ex_tree.getroot(), new_tree.getroot()]:
    #    print("im info")
    #    print(p_utils.extract_img_info(root))
    #for root in [ex_tree.getroot(), new_tree.getroot()]:
    #    _objects = [p_utils.extract_obj(obj) for obj in root.findall('object')]
    #    print("Len objects = %d" % len(_objects))
    #    print(_objects[0])
    #for root in [ex_tree.getroot(), new_tree.getroot()]:
    #    _relationships = [p_utils.extract_rel(rel) for rel in root.findall('relation')]
    #    print("Len rels = %d" % len(_relationships))
    #    print(_relationships[0])
