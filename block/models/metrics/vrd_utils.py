import torch
import numpy as np

def calculate_recall(R, tps, fps, scores, total_num_gts):
    pass

def eval_batch(dets_file, gts_file, num_dets=50, ov_thresh=0.5):
    pass

def eval_batch_union(dets_file, gts_file, num_dets=50, ov_thresh=0.5):
    pass

def computeOverlap(detBBs, gtBBs):
    pass

def computeArea(bb):
    pass

def computeIoU(bb1, bb2):
    pass

def getUnionBB(aBB, bBB):
    pass

def annot_to_gt(annot):
    pass

def item_to_det(item):
    pass
