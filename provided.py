"""
COMP 182 Homework 6 Solutions
"""

import math
import numpy
from collections import defaultdict


######################       PROVIDED CODE       ##########################

class HMM:
	"""
	Simple class to represent a Hidden Markov Model.
	"""

	def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
		self.order = order
		self.initial_distribution = initial_distribution
		self.emission_matrix = emission_matrix
		self.transition_matrix = transition_matrix


def read_pos_file(filename):
	"""
	Parses an input tagged text file.
	Input:
	filename --- the file to parse
	Returns: 
	The file represented as a list of tuples, where each tuple 
	is of the form (word, POS-tag).
	A list of unique words found in the file.
	A list of unique POS tags found in the file.
	"""
	file_representation = []
	unique_words = set()
	unique_tags = set()
	f = open(str(filename), "r")
	for line in f:
		if len(line) < 2 or len(line.split("/")) != 2:
			continue
		word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
		tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
		file_representation.append((word, tag))
		unique_words.add(word)
		unique_tags.add(tag)
	f.close()
	return file_representation, unique_words, unique_tags


#######################        MY CODE        ##########################

def compute_counts(training_data, order):
	token = len(training_data)
	dict_ctw = defaultdict(lambda: defaultdict(int))
	dict_ct = defaultdict(int)
	dict_ctt = defaultdict(int)
	if order == 3:
		dict_cttt = defaultdict(int)

	for i in range(len(training_data)):
		dict_ctw[training_data[i][1]][training_data[i][0]] += 1
		dict_ct[training_data[i][1]] += 1
		if i <= len(training_data) - 2:
			dict_ctt[(training_data[i][1], training_data[i+1][1])] += 1
		if order == 3 and i <= len(training_data) - 3:
			dict_cttt[(training_data[i][1], training_data[i+1][1], training_data[i+2][1])] += 1

	if order == 2:
		return token, dict_ctw, dict_ct, dict_ctt
	elif order == 3:
		return token, dict_ctw, dict_ct, dict_ctt, dict_cttt


