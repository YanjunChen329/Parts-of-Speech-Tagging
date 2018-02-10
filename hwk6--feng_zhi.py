"""
Comp182 Homework6 Part-of-speech tagging
"""
### provided code
import math
# import types
import pylab
import numpy
from collections import defaultdict


class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """

    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix
    def get_order(self):
        return self.order
    def get_initial(self):
        return self.initial_distribution
    def get_emission(self):
        return self.emission_matrix
    def get_transition(self):
        return self.transition_matrix


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

### student's code
def read_untagged_file(filename):
    """
    read the given untagged document
    :param filename: a ducument with bunch of words
    :return: a list of strings representing the given document
    """
    f = open(str(filename), "r")
    f = f.read()
    document = f.split()
    return document

def divide_sentences(document):
    """
    divide the given document into lists of sentences
    :param document: a list of strings representing the given document
    :return: a list of lists where each list is a list of strings representing a sentence
    """
    sentences = []
    sentence = []
    for word in document:
        sentence.append(word)
        if word == ".":
            sentences.append(sentence)
            sentence = []
    return sentences

def compute_counts(training_data, order):
    """
    obtain all of the necessary counts from the training corpus
    :param training_data: a list of (word, POS-tag) pairs returned by function read_pos_file
    :param order: an integer representing the order of the HMM, either 2 or 3
    :return: if order = 2, a tuple containing the number of tokens in training_data, a dictionary that contains C(t_i,w_i) for every
    unique tag and unique word, (keys correspond to tags), a dictionary that contains C(t_i) as above, a dictoinary
    that contains C(t_i-1, t_i) as above. If order = 3, in addition to these four above, it also returns a dictionary that
    contains C(t_i-2, t_i-1, t_i) as the fifth element.
    """
    #initialize a list to store output
    result = []
    #compute the number of tokens
    tokens = len(training_data)
    result.append(tokens)
    #compute counts
    taggings = defaultdict(lambda: defaultdict(int))
    tags = defaultdict(int)
    twoseq = defaultdict(lambda: defaultdict(int))
    for data in training_data:
        word = data[0]
        tag = data[1]
        tags[tag] += 1
        taggings[tag][word] += 1
    for idx in range(tokens - 1):
        data1 = training_data[idx]
        tag1 = data1[1]
        data2 = training_data[idx+1]
        tag2 = data2[1]
        twoseq[tag1][tag2] += 1
    result.append(taggings)
    result.append(tags)
    result.append(twoseq)
    # additional count if order is 3
    if order == 3:
        threeseq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for idx in range(tokens - 2):
            data1 = training_data[idx]
            data2 = training_data[idx+1]
            data3 = training_data[idx+2]
            tag1 = data1[1]
            tag2 = data2[1]
            tag3 = data3[1]
            threeseq[tag1][tag2][tag3] += 1
        result.append(threeseq)
    return tuple(result)

# train = read_pos_file("training copy.txt")[0]
# print compute_counts(train, 2)
# train1 = read_pos_file("testdata_tagged.txt")[0]
# print compute_counts(train1, 2)[0]
# print compute_counts(train1, 2)[1]
# print compute_counts(train1, 2)[2]
# print compute_counts(train1, 2)[3]
# print compute_counts(train1, 3)[4]

def compute_initial_distribution(training_data, order):
    """
    compute the initial distribution of the given data according to the order of HMM
    :param training_data: a list of (word, POS-tag) pairs returned by function read_pos_file
    :param order: an integer representing the order of the HMM, either 2 or 3
    :return: returns a dictionary containing the initial distribution of the given data according to the order of HMM
    """
    #compute the number of sentences in the given data
    sentences = 0
    for data in training_data:
        if data[1] == ".":
            sentences += 1
    if order == 2:
        dis = defaultdict(int)
        dis[training_data[0][1]] += 1
        for idx in range(len(training_data) - 1):
            data = training_data[idx]
            tag = data[1]
            if tag == ".":
                start = training_data[idx + 1][1]
                # if start != '.':
                dis[start] += 1
        for tag in dis.keys():
            dis[tag] /= float(sentences)
    elif order == 3:
        dis = defaultdict(lambda: defaultdict(int))
        first = training_data[0][1]
        second = training_data[1][1]
        if second != ".":
            dis[first][second] += 1
        for idx in range(len(training_data) - 2):
            data = training_data[idx]
            tag = data[1]
            if tag == ".":
                start1 = training_data[idx + 1][1]
                start2 = training_data[idx + 2][1]
                # if start1 != "." and start2 != '.':
                dis[start1][start2] += 1
        for tag1 in dis.keys():
            for tag2 in dis[tag1].keys():
                dis[tag1][tag2] /= float(sentences)
    return dis

# print compute_initial_distribution(train1, 2)
# print compute_initial_distribution(train1, 3)

def compute_emission_probabilities(unique_words, unique_tags, W, C):
    """
    compute the emission probabilities of each tag
    :param unique_words: second output set returned by read_pos_file, a set of all unique words in the given training corpus
    :param unique_tags: third output set returned by read_pos_file, a set of all unique tags in the given training corpus
    :param W: taggings dictionary computed by compute_counts
    :param C: tags dictionary computed by compute_counts
    :return: an emission matrix as a dictionary whose keys are the tags
    """
    prob = defaultdict(lambda: defaultdict(int))
    for word in unique_words:
        for tag in unique_tags:
            prob[tag][word] = W[tag][word] / float(C[tag])
    return prob


# print compute_emission_probabilities(read_pos_file("testdata_tagged.txt")[1],read_pos_file("testdata_tagged.txt")[2], compute_counts(train1, 2)[1], compute_counts(train1, 2)[2])

def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
    """
    compute the lambdas using the given algorithm
    :param unique_tags: a set of all of the unique tags in the given training corpus
    :param num_tokens: an integer representing the total number of tokens in the given training corpus
    :param C1: dictionary with count C(t_i)
    :param C2: dictionary with count C(t_i-1, t_i)
    :param C3: dictionary with count C(t_i-2, t_i-1, t_i)
    :param order: an integer representing the order of the HMM
    :return: a list that contains the three lambdas
    """
    if order == 3:
        lambdas = [0,0,0]
        for ti_2 in unique_tags:
            for ti_1 in unique_tags:
                for ti in unique_tags:
                    if C3[ti_2][ti_1][ti] > 0:
                        a = [0,0,0]
                        if num_tokens != 0:
                            a[0] = (C1[ti]-1) / float(num_tokens)
                        if C1[ti_1] != 1:
                            a[1] = (C2[ti_1][ti]-1) / float(C1[ti_1]-1)
                        if C2[ti_2][ti_1] != 1:
                            a[2] = (C3[ti_2][ti_1][ti]-1) / float(C2[ti_2][ti_1]-1)
                        amax = max(a)
                        for i in range(len(a)):
                            if a[i] == amax:
                                lambdas[i] += C3[ti_2][ti_1][ti]
                                break
    elif order == 2:
        lambdas = [0,0,0]
        for ti_1 in unique_tags:
            for ti in unique_tags:
                if C2[ti_1][ti] > 0:
                    a = [0,0]
                    if num_tokens != 0:
                        a[0] = (C1[ti] - 1) / float(num_tokens)
                    if C1[ti_1] != 1:
                        a[1] = (C2[ti_1][ti] - 1) / float(C1[ti_1] - 1)
                    amax = max(a)
                    for i in range(len(a)):
                        if a[i] == amax:
                            lambdas[i] += C2[ti_1][ti]
                            break
    summ = sum(lambdas)
    for idx in range(len(lambdas)):
        lambdas[idx] /= float(summ)
    return lambdas

# tokensnum = compute_counts(train1, 2)[0]
# words = read_pos_file("testdata_tagged.txt")[1]
# tags = read_pos_file("testdata_tagged.txt")[2]
# c1 = compute_counts(train1, 2)[2]
# c2 = compute_counts(train1, 2)[3]
# c3 = compute_counts(train1, 3)[4]
# print compute_lambdas(tags, tokensnum, c1, c2, c3, 2)
# print compute_lambdas(tags, tokensnum, c1, c2, c3, 3)

def compute_transition_probabilities(unique_tags, num_tokens, C1, C2, C3, lambdas, order):
    """
    build a transition probabilities matrix
    :param unique_tags: a set of all of the unique tags in the given training corpus
    :param num_tokens: an integer representing the total number of tokens in the given training corpus
    :param C1: dictionary with count C(t_i)
    :param C2: dictionary with count C(t_i-1, t_i)
    :param C3: dictionary with count C(t_i-2, t_i-1, t_i)
    :param order: an integer representing the order of the HMM
    :param lambdas: a list containing the lambdas needed in the given order
    :return: a dictionary representing the transition probabilities matrix of the given training corpus
    """
    if order == 2:
        trans = defaultdict(lambda:defaultdict(int))
        for previous in unique_tags:
            for current in unique_tags:
                trans[previous][current] = lambdas[0]*C1[current]/float(num_tokens) + lambdas[1]*C2[previous][current]/float(C1[previous])
        # trans.pop(".")
    elif order == 3:
        trans = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for early in unique_tags:
            for previous in unique_tags:
                for current in unique_tags:
                    if C2[early][previous] != 0:
                        trans[early][previous][current] = lambdas[0]*C1[current]/float(num_tokens) + lambdas[1]*C2[previous][current]/float(C1[previous]) + lambdas[2]*C3[early][previous][current]/float(C2[early][previous])
                    else:
                        trans[early][previous][current] = lambdas[0] * C1[current] / float(num_tokens) + lambdas[1] * C2[previous][current] / float(C1[previous])
        # trans.pop(".")
        # for early in trans:
        #     if "." in trans[early]:
        #         trans[early].pop(".")
    return trans

def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
    """
    build an hmm model using the given data
    :param training_data: a list of (word, POS-tag) pairs returned by function read_pos_file
    :param unique_tags: a set of all unique tags in the given training corpus
    :param unique_words: a set of all unique words in the given training corpus
    :param order: an integer representing the order of the HMM we want to build
    :param use_smoothing: a boolean value representing whether to use smoothing or not
    :return: an instance of the class hmm representing a fully trained hmm
    """
    initial = compute_initial_distribution(training_data, order)
    emission = compute_emission_probabilities(unique_words, unique_tags, compute_counts(training_data, order)[1], compute_counts(training_data, order)[2])
    num_tokens = compute_counts(training_data, order)[0]
    c1 = compute_counts(training_data, order)[2]
    c2 = compute_counts(training_data, order)[3]
    c3 = compute_counts(training_data, 3)[4]
    if use_smoothing:
        lambdas = compute_lambdas(unique_tags, num_tokens, c1, c2, c3, order)
    elif not use_smoothing:
        if order == 2:
            lambdas = [0,1,0]
        elif order ==3:
            lambdas = [0,0,1]
    transition = compute_transition_probabilities(unique_tags, num_tokens, c1, c2, c3, lambdas, order)
    hmm = HMM(order, initial, emission, transition)
    return hmm

# h1 = build_hmm(train1, tags, words, 2, True)
# h2 = build_hmm(train1, tags, words, 3, True)
# h3 = build_hmm(train1, tags, words, 2, False)
# h4 = build_hmm(train1, tags, words, 3, False)
# training_data_new = read_pos_file('training copy.txt')[0]
# unique_words = read_pos_file('training copy.txt')[1]
# unique_tags = read_pos_file('training copy.txt')[2]
# new_hmm = build_hmm(training_data_new, unique_tags, unique_words, 2, True)
# print new_hmm.get_initial()
# print new_hmm.get_emission()
# print new_hmm.get_transition()

def bigram_viterbi(hmm, sentence):
    """
    an implementation of the Viterbi Algorithm for the bigram model
    :param hmm: an instance of the class HMM representing a second order hidden markov model
    :param sentence: a list of word and a period at the end representing a sequence of observations
    :return: a list of (word, tag) tuples representing the tagged sentence
    """
    v = defaultdict(lambda: defaultdict(int))
    bp = defaultdict(lambda: defaultdict(int))
    initial = hmm.initial_distribution
    emission = hmm.emission_matrix
    transition = hmm.transition_matrix
    tags = emission.keys()
    for tag in tags:
        v[tag][0] = log(initial[tag]) + log(emission[tag][sentence[0]])
    for i in range(1, len(sentence)):
        for tag in emission.keys():
            max = v[tags[0]][i-1] + log(transition[tags[0]][tag])
            argmax = tags[0]
            for idx in range(1, len(tags)):
                if v[tags[idx]][i-1] + log(transition[tags[idx]][tag]) > max:
                    max = v[tags[idx]][i-1] + log(transition[tags[idx]][tag])
                    argmax = tags[idx]
            v[tag][i] = log(emission[tag][sentence[i]]) + max
            bp[tag][i] = argmax
    z = []
    zmax = v[tags[0]][len(sentence)-1]
    argzmax = tags[0]
    for idx in range(1,len(tags)):
        if v[tags[idx]][len(sentence)-1] > zmax:
            zmax = v[tags[idx]][len(sentence)-1]
            argzmax = tags[idx]
    z.append(argzmax)
    for i in range(len(sentence)-2, -1, -1):
        item = bp[z[0]][i+1]
        z.insert(0,item)
    result = []
    for idx in range(len(sentence)):
        result.append((sentence[idx],z[idx]))
    return result

def log(num):
    """
    compute the log of the given number
    :param num: a number
    :return: the log of num on base 2
    """
    if num == 0:
        return -float('inf')
    else:
        return math.log(float(num),2)

def update_hmm(hmm, sentence):
    """
    update the given hmm so that it includes the word who is in the test data but not in the training data
    :param hmm: an instance of the class HMM
    :param sentence: a list of words representing the given sentence of the test data
    :return: an updated hmm which includes the word who is in the test data but not in the training data
    """
    words = set([])
    tags = hmm.emission_matrix.keys()
    for tag in tags:
        for word in hmm.emission_matrix[tag].keys():
            words.add(word)
    unincluded = set([])
    for word in sentence:
        if word not in words:
            unincluded.add(word)
    if len(unincluded) != 0:
        for tag in tags:
            for word in hmm.emission_matrix[tag]:
                if hmm.emission_matrix[tag][word] != 0:
                    hmm.emission_matrix[tag][word] += 0.00001
    for tag in tags:
        for word in unincluded:
            hmm.emission_matrix[tag][word] = 0.00001
    for tag in tags:
        total = sum(hmm.emission_matrix[tag].values())
        for word in hmm.emission_matrix[tag]:
            hmm.emission_matrix[tag][word] /= float(total)
    return hmm

def trigram_viterbi(hmm, sentence):
    """
    an implementation of the Viterbi Algorithm for the bigram model
    :param hmm: an instance of the class HMM representing a third order hidden markov model
    :param sentence: a list of word and a period at the end representing a sequence of observations
    :return: a list of (word, tag) tuples representing the tagged sentence
    """
    v = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    initial = hmm.initial_distribution
    emission = hmm.emission_matrix
    transition = hmm.transition_matrix
    tags = emission.keys()
    for i in tags:
        for j in tags:
            v[i][j][1] = log(initial[i][j]) + log(emission[i][sentence[0]]) + log(emission[j][sentence[1]])
    for k in range(2, len(sentence)):
        for i in tags:
            for j in tags:
                max = float('-inf')
                argmax = tags[0]
                for g in tags:
                    if v[g][i][k-1] + log(transition[g][i][j]) > max:
                        max = v[g][i][k-1] + log(transition[g][i][j])
                        argmax = g
                v[i][j][k] = log(emission[j][sentence[k]]) + max
                bp[i][j][k] = argmax
    z = []
    max = float('-inf')
    arg1 = tags[0]
    arg2 = tags[0]
    for i in tags:
        for j in tags:
            if v[i][j][len(sentence)-1] > max:
                max = v[i][j][len(sentence)-1]
                arg1 = i
                arg2 = j
    z.append(arg1)
    z.append(arg2)
    for k in range(len(sentence)-1, 1, -1):
        item = bp[z[0]][z[1]][k]
        z.insert(0, item)
    result = []
    for idx in range(len(z)):
        result.append((sentence[idx], z[idx]))
    return result

real_train = read_pos_file("training.txt")[0]
words = read_pos_file("training.txt")[1]
tags = read_pos_file("training.txt")[2]
# train_hmm = build_hmm(real_train, tags, words, 2, True)
# train_hmm_tri = build_hmm(real_train, tags, words, 3, True)
# sen2 = ["The", "New", "Deal", "was", "a", "series", "of", "domestic", "programs", "enacted", "in", "the", "United", "States", "between", "1933", "and", "1936", ",", "and", "a", "few", "that", "came", "later", "."]
# print "bigram is", bigram_viterbi(train_hmm, sent)
# print "trigram is", trigram_viterbi(train_hmm_tri, sen2)

def cut_data(training_data, percent):
    """
    cut the given training data to get the first given percent of it
    :param training_data: a list of (word, POS-tag) pairs returned by function read_pos_file
    :param percent: an integer [0,100] representing the first certain percent we want to get from the data
    :return: a list of (word, POS-tag) pairs representing the first certain percent of the training data
    """
    cutpoint = int(round(percent/float(100) * len(training_data)))
    data = training_data[:cutpoint]
    tags = set([])
    words = set([])
    for pair in data:
        words.add(pair[0])
        tags.add(pair[1])
    tags = list(tags)
    words = list(words)
    return data, tags, words

def compute_accuracy(actual_tagging, result_tagging):
    """
    compute the percentage of all tags in the test set that agree with the tags produced by my HMM
    :param actual_tagging: a list of tuples representing the actual tagging of the given sentence
    :param result_tagging: a list of tuples representing the result tagging I got of the given sentence
    :return: the percentage of all tags inthe rest set that agree with the tags produced by my HMM
    """
    correct = 0
    for idx in range(len(actual_tagging)):
        if actual_tagging[idx] == result_tagging[idx]:
            correct += 1
    return correct /float(len(actual_tagging))

def run_experiment(hmm, test_filename, tagged_filename, order):
    """
     run an experiment using the document in the file as test data and return the accuracy
    :param hmm: an instance of the class HMM representing the given hmm
    :param filename: the name of a file containing the test data
    :param order: the order on which to perform the viterbi algorithm
    :param tagged_filename:  the name of the file containing the tagged data as the ground truth
    :return: the accuracy of the experiment
    """
    document = read_untagged_file(test_filename)
    hmm = update_hmm(hmm, document)
    sentences = divide_sentences(document)
    results = []
    for sentence in sentences:
        if order == 2:
            results.extend(bigram_viterbi(hmm, sentence))
        elif order == 3:
            results.extend(trigram_viterbi(hmm, sentence))
    truth = read_pos_file(tagged_filename)[0]
    accuracy = compute_accuracy(truth, results)
    return accuracy

# training1 = cut_data(real_train,1)[0]
# training5 = cut_data(real_train, 5)[0]
# training10 = cut_data(real_train, 10)[0]
# training25 = cut_data(real_train, 25)[0]
# training50 = cut_data(real_train, 50)[0]
# training75 = cut_data(real_train, 75)[0]

# tags1 = cut_data(real_train,1)[1]
# tags5 = cut_data(real_train,5)[1]
# tags10 = cut_data(real_train,10)[1]
# tags25 = cut_data(real_train,25)[1]
# tags50 = cut_data(real_train,50)[1]
# tags75 = cut_data(real_train,75)[1]

# words1 = cut_data(real_train,1)[2]
# words5 = cut_data(real_train,5)[2]
# words10 = cut_data(real_train,10)[2]
# words25 = cut_data(real_train,25)[2]
# words50 = cut_data(real_train,50)[2]
# words75 = cut_data(real_train,75)[2]

# nonbi1 = build_hmm(training1, tags1, words1, 2, False)
# nonbi5 = build_hmm(training5, tags5, words5, 2, False)
# nonbi10 = build_hmm(training10, tags10, words10, 2, False)
# nonbi25 = build_hmm(training25, tags25, words25, 2, False)
# nonbi50 = build_hmm(training50, tags50, words50, 2, False)
# nonbi75 = build_hmm(training75, tags75, words75, 2, False)
# nonbi100 = build_hmm(real_train, tags, words, 2, False)

# exp1 = [nonbi1, nonbi5, nonbi10, nonbi25, nonbi50, nonbi75, nonbi100]

# smobi1 = build_hmm(training1, tags1, words1, 2, True)
# smobi5 = build_hmm(training5, tags5, words5, 2, True)
# smobi10 = build_hmm(training10, tags10, words10, 2, True)
# smobi25 = build_hmm(training25, tags25, words25, 2, True)
# smobi50 = build_hmm(training50, tags50, words50, 2, True)
# smobi75 = build_hmm(training75, tags75, words75, 2, True)
smobi100 = build_hmm(real_train, tags, words, 2, True)

# exp3 = [smobi1, smobi5, smobi10, smobi25, smobi50, smobi75, smobi100]

# nontri1 = build_hmm(training1, tags1, words1, 3, False)
# nontri5 = build_hmm(training5, tags5, words5, 3, False)
# nontri10 = build_hmm(training10, tags10, words10, 3, False)
# nontri25 = build_hmm(training25, tags25, words25, 3, False)
# nontri50 = build_hmm(training50, tags50, words50, 3, False)
# nontri75 = build_hmm(training75, tags75, words75, 3, False)
# nontri100 = build_hmm(real_train, tags, words, 3, False)

# exp2 = [nontri1, nontri5, nontri10, nontri25, nontri50, nontri75, nontri100]

# smotri1 = build_hmm(training1, tags1, words1, 3, True)
# smotri5 = build_hmm(training5, tags5, words5, 3, True)
# smotri10 = build_hmm(training10, tags10, words10, 3, True)
# smotri25 = build_hmm(training25, tags25, words25, 3, True)
# smotri50 = build_hmm(training50, tags50, words50, 3, True)
# smotri75 = build_hmm(training75, tags75, words75, 3, True)
# smotri100 = build_hmm(real_train, tags, words, 3, True)

# exp4 = [smotri1, smotri5, smotri10, smotri25, smotri50, smotri75, smotri100]

# print 'nonsmoothing', run_experiment(nonbi100, "testdata_untagged.txt", "testdata_tagged.txt", 2)
print run_experiment(smobi100, "testdata_untagged.txt", "testdata_tagged.txt", 2)
# print run_experiment(nontri100, "testdata_untagged.txt", "testdata_tagged.txt", 3)
# print run_experiment(smotri1, "testdata_untagged.txt", "testdata_tagged.txt", 3)
# print 'smoothing', run_experiment(smobi100,"testdata_untagged.txt", "testdata_tagged.txt", 2)
# doc = read_untagged_file("testdata_untagged.txt")
# truth = read_pos_file("testdata_tagged.txt")[0]
# result = bigram_viterbi(smobi100, doc)
# print result
# print compute_accuracy(truth, result)

def make_plot(exp1, exp2, exp3, exp4):
    """
    Plot a line graph with the provided data and store it under the given file
    :param experiments1,2,3,4: four lists of hmms representing the hmms used for each experiments
    :param filename: the location to store the plot
    :return: a line graph
    """
    percentage = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1]
    data = []
    result1 = {}
    for idx in range(len(exp1)):
        result1[percentage[idx]] = run_experiment(exp1[idx], "testdata_untagged.txt", "testdata_tagged.txt", 2)
    data.append(result1)
    result2 = {}
    for idx in range(len(exp2)):
        result2[percentage[idx]] = run_experiment(exp2[idx], "testdata_untagged.txt", "testdata_tagged.txt", 3)
    data.append(result2)
    result3 = {}
    for idx in range(len(exp3)):
        result3[percentage[idx]] = run_experiment(exp3[idx], "testdata_untagged.txt", "testdata_tagged.txt", 2)
    data.append(result3)
    result4 = {}
    for idx in range(len(exp1)):
        result4[percentage[idx]] = run_experiment(exp4[idx], "testdata_untagged.txt", "testdata_tagged.txt", 3)
    data.append(result4)
    title = "HMM Experiments Results"
    xlabel = "Percentage of Training Corpus"
    ylabel = "Accuracy"
    labels = ["Experiment1","Experiment2","Experiment3","Experiment4"]
    filename = "/Users/edwardfeng/Desktop/Spring 2017/Comp 182/HW6/HMM Experiments Results"
    plot_lines(data, title, xlabel, ylabel, labels, filename)

# make_plot(exp1, exp2, exp3, exp4)









#for test
# initial = defaultdict(int)
# initial['E'] = 1
# emission = defaultdict(lambda: defaultdict(int))
# emission['E']['a'] = 0.25
# emission['E']['b'] = 0.25
# emission['E']['c'] = 0.25
# emission['E']['d'] = 0.25
# emission['5']['a'] = 0.05
# emission['5']['c'] = 0.95
# emission['I']['a'] = 0.4
# emission['I']['b'] = 0.1
# emission['I']['c'] = 0.1
# emission['I']['d'] = 0.4
# emission['.']['.'] = 1
# transition = defaultdict(lambda: defaultdict(int))
# transition['E']['E'] = 0.9
# transition['E']['5'] = 0.1
# transition['5']['I'] = 1
# transition['I']['I'] = 0.9
# transition['I']['.'] = 0.1
#
# test_hmm = HMM(2, initial, emission, transition)
# test_sentence = ['a', 'c', 'c', 'a', '.']
# print bigram_viterbi(test_hmm, test_sentence)
# 0.9681724846