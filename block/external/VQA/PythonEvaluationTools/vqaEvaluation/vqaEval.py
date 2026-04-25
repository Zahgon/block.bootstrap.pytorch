# coding=utf-8

__author__='aagrawal'

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
import sys
import re

class VQAEval:
	def __init__(self, vqa, vqaRes, n=2):
		raise NotImplementedError

	
	def evaluate(self, quesIds=None):
		"""
		quesIds: a list of question ids on which you want to compute the 
		evaluation. Typically, if your predictions is not carried on the 
		whole dataset, just pass the list of question ids for which you 
		have an answer here.
		"""
		raise NotImplementedError
		#print "Done computing accuracy"
	
	def processPunctuation(self, inText):
		raise NotImplementedError
	
	def processDigitArticle(self, inText):
		raise NotImplementedError

	def setAccuracy(self, accQA, accQuesType, accAnsType):
		raise NotImplementedError
			
	def setEvalQA(self, quesId, acc):
		raise NotImplementedError

	def setEvalQuesType(self, quesId, quesType, acc):
		raise NotImplementedError
	
	def setEvalAnsType(self, quesId, ansType, acc):
		raise NotImplementedError

	def updateProgress(self, progress):
		raise NotImplementedError
		# sys.stdout.write(text)
		# sys.stdout.flush()

