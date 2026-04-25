import argparse
import json
import random
import os
from os.path import join
import sys
from bootstrap.lib.logger import Logger
from block.external.VQA.PythonHelperTools.vqaTools.vqa import VQA
from block.external.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

def real_split_name(split):
    raise NotImplementedError

def main(dir_vqa, dir_exp, dir_rslt, epoch, split, cmd_line=True, logs_name="logs", rm=True):
    raise NotImplementedError


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_vqa',   type=str, default='/local/cadene/data/vqa')
    parser.add_argument('--dir_exp', type=str, default='logs/16_12_13_20:39:55/')
    parser.add_argument('--dir_rslt', type=str, default='logs/16_12_13_20:39:55/results/train/epoch,1')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--split',  type=str, default='train')
    parser.add_argument('--logs_name',  type=str, default='logs')
    parser.add_argument('--rm',  type=int, default=1)
    args = parser.parse_args()

    main(args.dir_vqa, args.dir_exp, args.dir_rslt, args.epoch, args.split, logs_name=args.logs_name, rm=args.rm)
    
    #json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    #json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    #json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))
