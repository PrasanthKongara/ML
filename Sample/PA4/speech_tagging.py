import numpy as np
import time
import random
from hmm import HMM
from collections import Counter



def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data
    
    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags
    
    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    start=[]
    obs = []
    for i in range(len(train_data)):
        start.append(train_data[i].tags[0])
        obs+= train_data[i].words
    obs=list(set(obs))
    A=np.zeros([len(tags),len(tags)])
    B=np.zeros([len(tags),len(obs)])
    for i in range(len(train_data)):
        words_ = train_data[i].words
        tags_ = train_data[i].tags
        for j in range(len(tags_)):
            if j < len(tags_)-1:
                A[tags.index(tags_[j])][tags.index(tags_[j+1])] += 1
            B[tags.index(tags_[j])][obs.index(words_[j])] +=1
    A_d = np.sum(A,axis=1,keepdims=True)
    A_d[A_d==0] = 1
    B_d = np.sum(B,axis=1,keepdims=True)
    B_d[B_d==0] = 1
    A = A/A_d 
    B = B/B_d
           
    count=Counter(start)
    pi=np.zeros(len(tags))
    state_dict={}
    for i in range(len(tags)):
        if tags[i] in count:
            pi[i]=count[tags[i]]/len(start)
            
    obs_dict={k: v for v, k in enumerate(obs)}
    state_dict={k: v for v, k in enumerate(tags)}
    model = HMM(pi, A, B, obs_dict, state_dict)
    ###################################################
    # Edit here
    ###################################################
    return model


# TODO:
def speech_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class
    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    temp=[]
    for i in range(len(test_data)):
        temp+=test_data[i].words
    obs=list(set(temp))
    k=len(model.obs_dict)
    n=np.ones((len(model.state_dict),1))*1e-6
    for i in range(len(obs)):
        if obs[i] not in model.obs_dict:
            model.obs_dict[obs[i]]=k
            model.B=np.append(model.B,n,axis=1)
    for i in range(len(test_data)):
        tagging.append(model.viterbi(test_data[i].words))
    ###################################################
    # Edit here
    ###################################################
    return tagging

