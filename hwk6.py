import math
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

	def update_emission(self, emission_matrix):
		self.emission_matrix = emission_matrix


def read_pos_file(filename):
	"""
	Parses an input tagged text file.
	Input:
	filename --- the file to parse
	Returns:
	The file represented as a list of tuples, where each tuple is of the form (word, POS-tag).
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

# training = read_pos_file('training.txt')[0]
# uniquewords = read_pos_file('training.txt')[1]
# uniquetags = read_pos_file('training.txt')[2]


#######################        MY CODE        ##########################

def compute_counts(training_data, order):
	"""
	Count the number of tokens, dictionaries C(ti, wi), C(ti), C(ti-1, ti) and C(ti-2, ti-1, ti)
	if order = 3 from the training data.

	Arguments:
	training_data -- a list of (word, tag) pairs used for training
	order -- 2: bigram; 3: trigram

	Return:
	the number of tokens, C(ti, wi), C(ti), C(ti-1, ti) and C(ti-2, ti-1, ti) if order = 3
	"""
	# Order can only be 2 or 3
	if order != 2 and order != 3:
		raise Exception('order should either be 2 or 3')

	# Initialize counting dictionaries
	token = len(training_data)
	dict_ctw = defaultdict(lambda: defaultdict(int))
	dict_ct = defaultdict(int)
	dict_ctt = defaultdict(lambda: defaultdict(int))
	if order == 3:
		dict_cttt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

	# Iterate over every (word, tag) pair in the training data
	for i in range(len(training_data)):
		dict_ctw[training_data[i][1]][training_data[i][0]] += 1
		dict_ct[training_data[i][1]] += 1
		# The if statement prevents going out of index
		if i <= len(training_data) - 2:
			dict_ctt[training_data[i][1]][training_data[i+1][1]] += 1
		if order == 3 and i <= len(training_data) - 3:
			dict_cttt[training_data[i][1]][training_data[i+1][1]][training_data[i+2][1]] += 1

	if order == 2:
		return token, dict_ctw, dict_ct, dict_ctt
	else:
		return token, dict_ctw, dict_ct, dict_ctt, dict_cttt

# print training
# print compute_counts(training, 2)
# print compute_counts(training, 3)


def compute_initial_distribution(training_data, order):
	"""
	Compute the initial probability distribution as a dictionary that contains pi1
	if order equals 2, and pi2 if order equals 3.

	Arguments:
	training_data -- a list of (word, tag) pairs used for training
	order -- 2: bigram; 3: trigram

	Return:
	the initial probability distribution as a dictionary
	"""
	# Calculate pi1 if order = 2
	if order == 2:
		counter1 = 0
		biini_pi = defaultdict(int)
		# Iterate over every (word, tag) pair in the training data
		for i in range(len(training_data)):
			# We only count the tags showing up after a period
			if training_data[i][1] == '.' and i <= len(training_data) - 2:
				biini_pi[training_data[i+1][1]] += 1
				counter1 += 1
		# Transform the counting into probabilities
		for tag in biini_pi:
			biini_pi[tag] = float(biini_pi[tag]) / counter1

		return biini_pi

	# Calculate pi2 if order = 3
	elif order == 3:
		counter2 = 0
		triini_pi = defaultdict(lambda: defaultdict(int))
		# Iterate over every (word, tag) pair in the training data
		for i in range(len(training_data)):
			# We only count the word showing up after a period
			if training_data[i][1] == '.' and i <= len(training_data) - 3:
				triini_pi[training_data[i+1][1]][training_data[i+2][1]] += 1
				counter2 += 1
		# Transform the counting into probabilities
		for tag1 in triini_pi:
			for tag2 in triini_pi[tag1]:
				triini_pi[tag1][tag2] = float(triini_pi[tag1][tag2]) / counter2

		return triini_pi

	# Raise an error if the order is not 2 or 3
	else:
		raise Exception('order should either be 2 or 3')

# print compute_initial_distribution(training, 2)
# print compute_initial_distribution(training, 3)


def compute_emission_probabilities(unique_words, unique_tags, W, C):
	"""
	Compute the emission probability matrix as a dictionary whose keys are the tags.

	Arguments:
	unique_words -- a set of words that appeared in the training data
	unique_tags -- a set of tags that appeared in the training data
	W -- C(ti, wi)
	C -- C(ti)

	Return:
	the emission probability matrix
	"""
	# Initialize the matrix
	emission = defaultdict(lambda: defaultdict(float))
	# Iterate over every word that belong to certain tags
	for tag in W:
		for word in W[tag]:
			emission[tag][word] = float(W[tag][word]) / float(C[tag])
	return emission

# print compute_emission_probabilities(uniquewords, uniquetags, compute_counts(training, 2)[1],
# 									compute_counts(training, 2)[2])


def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
	"""
	Implement algorithm ComputeLambda to compute lambdas

	Arguments:
	unique_tags -- a set of tags that appeared in the training data
	num_tokens -- number of tokens in the training data
	C1 -- C(ti)
	C2 -- C(ti-1, ti)
	C3 -- C(ti-2, ti-1, ti)
	order -- 2: bigram; 3: trigram

	Return:
	three lambda values for the linear interpolation
	"""
	lam0, lam1, lam2 = 0, 0, 0
	if order == 2:
		# Iterate over every bigram ti-1, ti with C(ti-1, ti) > 0
		for tag1 in C2:
			for tag2 in C2[tag1]:
				if C2[tag1][tag2] > 0:
					# Calculate the two alpha
					alpha0 = float(C1[tag2] - 1) / num_tokens
					if C1[tag1] - 1 == 0:
						alpha1 = 0
					else:
						alpha1 = float(C2[tag1][tag2] - 1) / float(C1[tag1] - 1)

					# Modify the lambda based on which alpha has the highest value
					if alpha0 == max([alpha0, alpha1]):
						lam0 += C2[tag1][tag2]
					else:
						lam1 += C2[tag1][tag2]

		# Normalize the three lambda values to let them sum up to 1
		sumlam = float(lam0 + lam1)
		lam0 /= sumlam
		lam1 /= sumlam

	elif order == 3:
		# Iterate over every trigram ti-2, ti-1, ti with C(ti-2, ti-1, ti) > 0
		for tag1 in C3:
			for tag2 in C3[tag1]:
				for tag3 in C3[tag1][tag2]:
					if C3[tag1][tag2][tag3] > 0:

						# Calculate the three alpha
						alpha0 = float(C1[tag3] - 1) / num_tokens
						if C1[tag2] - 1 == 0:
							alpha1 = 0
						else:
							alpha1 = float(C2[tag2][tag3] - 1) / float(C1[tag2] - 1)
						if C2[tag1][tag2] - 1 == 0:
							alpha2 = 0
						else:
							alpha2 = float(C3[tag1][tag2][tag3] - 1) / float(C2[tag1][tag2] - 1)

						# Modify the lambda based on which alpha has the highest value
						if alpha0 == max([alpha0, alpha1, alpha2]):
							lam0 += C3[tag1][tag2][tag3]
						elif alpha1 == max([alpha0, alpha1, alpha2]):
							lam1 += C3[tag1][tag2][tag3]
						else:
							lam2 += C3[tag1][tag2][tag3]

		# Normalize the three lambda values to let them sum up to 1
		sumlam = float(lam0 + lam1 + lam2)
		lam0 /= sumlam
		lam1 /= sumlam
		lam2 /= sumlam

	else:
		raise Exception('order should either be 2 or 3')

	return lam0, lam1, lam2

# print compute_lambdas(uniquetags, compute_counts(training, 2)[0], compute_counts(training, 2)[2],
# 						compute_counts(training, 2)[3], None, 2)
# print compute_lambdas(uniquetags, compute_counts(training, 3)[0], compute_counts(training, 3)[2],
# 						compute_counts(training, 3)[3], compute_counts(training, 3)[4], 3)


def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
	"""
	Build a HMM based on the training data, order and whether or not we use smoothing

	Arguments:
	training_data -- the data used to train our HMM
	unique_tags -- set of all tags appearing in the training data
	unique_words -- set of all words appearing in the training data
	order -- 2: bigram; 3: trigram
	use_smoothing -- whether or not we use the smoothing method

	Return:
	an HMM built with the given data
	"""
	# Got the initial probability distribution and the emission matrix
	init_dis = compute_initial_distribution(training_data, order)
	emission_matrix = compute_emission_probabilities(unique_words, unique_tags,
													compute_counts(training_data, order)[1],
													compute_counts(training_data, order)[2])
	if order == 2:
		# Initialize the transition matrix and get all parameters ready
		token, Cw, C1, C2 = compute_counts(training_data, order)
		transition = defaultdict(lambda: defaultdict(float))
		# Got the lambda value for computing transition probabilities
		if use_smoothing:
			lam0, lam1, lam2 = compute_lambdas(unique_tags, token, C1, C2, None, order)
		else:
			lam0, lam1, lam2 = 0, 1, 0

		# Iterate over all pairs of states and calculate its probability
		for tag1 in emission_matrix.keys():
			for tag2 in emission_matrix.keys():
				# If any of the denominator equals to 0, let the term be zero
				if C1[tag1] != 0:
					transition[tag1][tag2] = lam1 * (float(C2[tag1][tag2]) / float(C1[tag1])) + \
											lam0 * (float(C1[tag2]) / token)
				else:
					transition[tag1][tag2] = lam0 * (float(C1[tag2]) / token)

	elif order == 3:
		# Initialize the transition matrix and get all parameters ready
		token, Cw, C1, C2, C3 = compute_counts(training_data, order)
		transition = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
		# Got the lambda value for computing transition probabilities
		if use_smoothing:
			lam0, lam1, lam2 = compute_lambdas(unique_tags, token, C1, C2, C3, order)
		else:
			lam0, lam1, lam2 = 0, 0, 1

		# Iterate over all permutations of three states and calculate its probability
		for tag1 in emission_matrix.keys():
			for tag2 in emission_matrix.keys():
				for tag3 in emission_matrix.keys():
					# If any of the denominator equals to 0, let the term be zero
					if C2[tag1][tag2] != 0:
						part1 = lam2 * (C3[tag1][tag2][tag3] / float(C2[tag1][tag2]))
					else:
						part1 = 0
					if C1[tag2] != 0:
						part2 = lam1 * (float(C2[tag2][tag3]) / float(C1[tag2]))
					else:
						part2 = 0
					transition[tag1][tag2][tag3] = part1 + part2 + lam0 * (float(C1[tag3]) / token)
	else:
		raise Exception('order should either be 2 or 3')

	return HMM(order, init_dis, emission_matrix, transition)

# tagging1 = build_hmm(training, uniquetags, uniquewords, 2, True)
# print tagging1.initial_distribution
# print tagging1.emission_matrix
# print tagging1.transition_matrix
# tagging2 = build_hmm(training, uniquetags, uniquewords, 3, True)
# print tagging2.initial_distribution
# print tagging2.emission_matrix
# print tagging2.transition_matrix


def update_hmm(hmm, new_words):
	"""
	Update the emission probability of the given HMM based on the given set of unseen words

	Arguments:
	hmm -- a HMM class object
	new_words -- a set of words that's not seen in the training data
	"""
	emission = hmm.emission_matrix
	total_prob = 0.0
	# Iterate over each tag in the matrix
	for tag in emission:
		# Add 0.0001 probability to every word in the emission matrix
		for word in emission[tag]:
			if emission[tag][word] != 0:
				emission[tag][word] += 0.00001
		# Give the new word 0.00001 emission probability
		for new_word in new_words:
			emission[tag][new_word] = 0.00001

		# Normalize the emission probability to make it 1
		total_prob += sum(emission[tag].values())
		for word2 in emission[tag]:
			emission[tag][word2] *= (1.0 / total_prob)
	# Update the new matrix
	hmm.update_emission(emission)


def log_hmm(hmm):
	"""
	Take the log probabilities of all the probability matrices in the given HMM.

	Arguments:
	hmm -- a HMM class object

	Return:
	A HMM with the log probabilities of the original HMM.
	"""
	# Get the three probability matrices from the original HMM
	init = hmm.initial_distribution
	emission = hmm.emission_matrix
	transit = hmm.transition_matrix

	if hmm.order == 2:
		# Log every entry of the initial probability distribution matrix
		log_init = defaultdict(lambda: float('-inf'))
		for tag in init:
			if init[tag] != 0:
				log_init[tag] = math.log10(init[tag])

		# Log every entry of the transitional probability matrix
		log_transit = defaultdict(lambda: defaultdict(lambda: float('-inf')))
		for tag1 in transit:
			for tag2 in transit[tag1]:
				if transit[tag1][tag2] != 0:
					log_transit[tag1][tag2] = math.log10(transit[tag1][tag2])

	elif hmm.order == 3:
		# Log every entry of the initial probability distribution matrix
		log_init = defaultdict(lambda: defaultdict(lambda: float('-inf')))
		for tag1 in init:
			for tag2 in init[tag1]:
				if init[tag1][tag2] != 0:
					log_init[tag1][tag2] = math.log10(init[tag1][tag2])

		# Log every entry of the transitional probability matrix
		log_transit = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float('-inf'))))
		for tag1 in transit:
			for tag2 in transit[tag1]:
				for tag3 in transit[tag1][tag2]:
					if transit[tag1][tag2][tag3] != 0:
						log_transit[tag1][tag2][tag3] = math.log10(transit[tag1][tag2][tag3])
	# The order has to be either 2 or 3
	else:
		raise Exception('order should either be 2 or 3')

	# Log every entry of the emission probability matrix
	log_emission = defaultdict(lambda: defaultdict(lambda: float('-inf')))
	for tag in emission:
		for word in emission[tag]:
			if emission[tag][word] != 0:
				log_emission[tag][word] = math.log10(emission[tag][word])

	return HMM(hmm.order, log_init, log_emission, log_transit)

# log_tagging1 = log_hmm(tagging1)
# # print log_tagging1.initial_distribution
# # print log_tagging1.emission_matrix
# print log_tagging1.transition_matrix
# log_tagging2 = log_hmm(tagging2)
# print log_tagging2.initial_distribution
# print log_tagging2.emission_matrix
# print log_tagging2.transition_matrix


def bigram_viterbi(hmm, sentence):
	"""
	Implement viterbi algorithm with the bigram model (1st order HMM).

	Arguments:
	hmm -- a HMM class object
	sentence -- a sentence that needs to be tagged

	Return:
	A list of (word, tag) pairs that maximize the probability
	"""
	# Log the probability matrices in HMM to avoid underflow
	loghmm = log_hmm(hmm)
	init = loghmm.initial_distribution
	emission = loghmm.emission_matrix
	transit = loghmm.transition_matrix
	# Initialize the scoring matrix and the traceback matrix
	p_matrix = defaultdict(lambda: defaultdict(lambda: float('-inf')))
	bp_matrix = defaultdict(lambda: defaultdict(lambda: ""))

	# Enter the initial probability distribution
	for tag in init:
		p_matrix[0][tag] = init[tag] + emission[tag][sentence[0]]

	# Iterate through every index of the sentence from index 1
	for i in range(1, len(sentence)):
		# Iterate over every state
		for tag2 in emission:
			max_value = float('-inf')
			max_tag = ""
			# Find the prior state that maximizes the probability
			for tag1 in emission:
				if p_matrix[i-1][tag1] + transit[tag1][tag2] > max_value:
					max_value = p_matrix[i-1][tag1] + transit[tag1][tag2]
					max_tag = tag1
			# Fill in both matrices
			p_matrix[i][tag2] = emission[tag2][sentence[i]] + max_value
			bp_matrix[i][tag2] = max_tag

	# Find the tag of the last word of sentence that maximizes the probability
	value = max(p_matrix[len(sentence) - 1].values())
	last_tag = [tag for tag, val in p_matrix[len(sentence) - 1].items() if val == value][0]
	# Add the (word, tag) pair into the sequence
	Z = []
	Z.append(tuple([sentence[-1], last_tag]))

	# Do traceback to find the "path" that maximizes the probability
	for j in range(len(sentence)-1, 0, -1):
		Z.append(tuple([sentence[j-1], bp_matrix[j][Z[len(sentence)-j-1][1]]]))

	return Z[::-1]

# print bigram_viterbi(tagging1, ['Yes', '!'])
# print bigram_viterbi(tagging1, ['I', 'fell', 'in', 'love', 'with', 'a', 'woman', '.'])
# print bigram_viterbi(tagging1, ['The', 'grand', 'jury', 'commented', 'on', 'a', 'number',
# 								'of', 'other', 'topics', '.'])
# print bigram_viterbi(tagging1, ['Mrs.', 'Shaefer', 'never', 'got', 'around', 'to', 'joining', '.'])
# print bigram_viterbi(tagging1, ['All', 'we', 'gotta', 'do', 'is', 'go', 'around', 'the', 'corner', '.'])
# print bigram_viterbi(tagging1, ['Chateau', 'Petrus', 'costs', 'around', '250', '.'])


def trigram_viterbi(hmm, sentence):
	"""
	Implement viterbi algorithm with the trigram model (2nd order HMM).

	Arguments:
	hmm -- a HMM class object
	sentence -- a sentence that needs to be tagged

	Return:
	A list of (word, tag) pairs that maximize the probability
	"""
	# The length of sentence has to be greater than or equal to 2 in order to use trigram
	if len(sentence) < 2:
		raise Exception("The length of sentence should be greater than 2")

	# Log the probability matrices in HMM to avoid underflow
	loghmm = log_hmm(hmm)
	init = loghmm.initial_distribution
	emission = loghmm.emission_matrix
	transit = loghmm.transition_matrix
	# Initialize the scoring matrix and the traceback matrix
	p_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float('-inf'))))
	bp_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "")))

	# Enter the initial probability distribution
	for tag1 in init:
		for tag2 in init[tag1]:
			p_matrix[1][tag1][tag2] = init[tag1][tag2] + emission[tag1][sentence[0]] + \
									emission[tag2][sentence[1]]

	# Iterate through every index of the sentence from index 2
	for i in range(2, len(sentence)):
		# Iterate over every pair of states
		for tag2 in emission:
			for tag3 in emission:
				max_value = float('-inf')
				max_tag1 = ""
				# Find the tag(happen before the pair) that maximizes probability
				for tag1 in emission:
					if p_matrix[i - 1][tag1][tag2] + transit[tag1][tag2][tag3] > max_value:
						max_value = p_matrix[i - 1][tag1][tag2] + transit[tag1][tag2][tag3]
						max_tag1 = tag1
				# Fill in both matrices
				p_matrix[i][tag2][tag3] = emission[tag3][sentence[i]] + max_value
				bp_matrix[i][tag2][tag3] = max_tag1

	# Find the pair of tags with the largest probability for the last two words in the sentence
	last_value = float('-inf')
	last_tag1 = ""
	last_tag2 = ""
	for tag1 in p_matrix[len(sentence)-1]:
		for tag2 in p_matrix[len(sentence)-1][tag1]:
			if p_matrix[len(sentence)-1][tag1][tag2] > last_value:
				last_value = p_matrix[len(sentence)-1][tag1][tag2]
				last_tag1 = tag1
				last_tag2 = tag2
	# Add the two (word, tag) pairs into the sequence
	Z = []
	Z.append(tuple([sentence[-1], last_tag2]))
	Z.append(tuple([sentence[-2], last_tag1]))

	# Do traceback to find the "path" that maximizes the probability
	for j in range(len(sentence) - 2, 0, -1):
		Z.append(tuple([sentence[j - 1],
						bp_matrix[j + 1][Z[len(sentence) - j - 1][1]][Z[len(sentence) - j - 2][1]]]))

	return Z[::-1]

# print trigram_viterbi(tagging2, ['Yes', '!'])
# print trigram_viterbi(tagging2, ['I', 'fell', 'in', 'love', 'with', 'a', 'girl', '.'])
# print trigram_viterbi(tagging2, ['The', 'grand', 'jury', 'commented', 'on', 'a', 'number',
# 								'of', 'other', 'topics', '.'])
# print trigram_viterbi(tagging2, ['Mrs.', 'Shaefer', 'never', 'got', 'around', 'to', 'joining', '.'])
# print trigram_viterbi(tagging2, ['All', 'we', 'gotta', 'do', 'is', 'go', 'around', 'the', 'corner', '.'])
# print trigram_viterbi(tagging2, ['Chateau', 'Petrus', 'costs', 'around', '250', '.'])


def generate_training_data(training_file, percentage):
	"""
	Generate the specific training data, unique words and unique tags based on the given percentage.

	Arguments:
	training_file: the file name of the original training data
	percentage: the percentage of data that we want

	Return
	The training data, unique words and unique tags after the percentage cut.
	"""
	# Read the training file and got the raw training data
	training_raw = read_pos_file(training_file)[0]
	training_data = []
	unique_words = set([])
	unique_tags = set([])

	# Select parts of the training data based on the given percentage
	for idx in range(int(len(training_raw) * percentage)):
		training_data.append(training_raw[idx])
		unique_words.add(training_raw[idx][0])
		unique_tags.add(training_raw[idx][1])

	return training_data, unique_words, unique_tags


def part_of_speech_tagging(training_data, words, tags, order, use_smoothing, testing_file, truth_file=None):
	"""
	Build the HMM based on the input training data and do Part-of-speech Tagging on the
	testing data. Return our prediction and its accuracy.

	Arguments:
	training_file -- the tagged file used to build HMM
	percentage -- percentage of training data we want to take
	order: 2 -- use the bigram model; 3: use the trigram model
	use_smoothing -- use smoothing or not
	testing_file -- the untagged file used to test
	truth_file -- the optional tagged file used to quantify accuracy

	Return:
	The tagging we got and the accuracy of our prediction
	"""
	# Build HMM from the parameters
	hmm = build_hmm(training_data, tags, words, order, use_smoothing)

	# Read the testing file
	text = open(testing_file, 'r').read()
	text = text.split(' ')

	# Search for new words in the file and update the HMM
	new_words = set([])
	for word in text:
		if word not in words:
			new_words.add(word)
	update_hmm(hmm, new_words)

	sentence = []  # sequence of sentence
	tagging = []  # sequence of tagging
	for word in text:
		sentence.append(word)
		# A sentence is formed and needs to be tagged whenever we saw a period
		if word == '.':
			if order == 2:
				tagging.extend(bigram_viterbi(hmm, sentence))
			else:
				tagging.extend(trigram_viterbi(hmm, sentence))
			sentence = []  # Proceed to the new sentence

	# If there is no tagged file, return the tagging directly
	if truth_file is None:
		return tagging

	truth = read_pos_file(truth_file)[0]
	accuracy = 0.0
	# Both tagging should have the same length
	if len(truth) != len(tagging):
		raise Exception("The tagged file does not correspond to the untagged file.")
	# Calculate the accuracy of the tagging produced by HMM
	for idx in range(len(truth)):
		if truth[idx] == tagging[idx]:
			accuracy += 1.0 / len(truth)

	return accuracy, tagging

# print part_of_speech_tagging(training, uniquewords, uniquetags, 2, True, 'testdata1.txt')
# print part_of_speech_tagging(training, uniquewords, uniquetags, 3, True, 'testdata1.txt')
# print part_of_speech_tagging(training, uniquewords, uniquetags, 2, True, 'testdata2.txt')
# print part_of_speech_tagging(training, uniquewords, uniquetags, 3, True, 'testdata2.txt')
# print part_of_speech_tagging(training, uniquewords, uniquetags,
# 							2, True, 'testdata_untagged.txt', 'testdata_tagged.txt')
# print part_of_speech_tagging(training, uniquewords, uniquetags,
# 							3, True, 'testdata_untagged.txt', 'testdata_tagged.txt')


def run():
	"""
	Run four experiments on the testing data and plot the accuracy curve.
	"""
	graph = []
	dic = {}
	param = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]  # percentage of data that we are going to analyze
	# Got the training data with different percentage
	training = {}
	for percent in param:
		training[percent] = tuple(generate_training_data('training.txt', percent))

	# Experiment 1
	for percent in param:
		dic[percent] = part_of_speech_tagging(training[percent][0], training[percent][1],
											training[percent][2], 2, False,
											'testdata_untagged.txt', 'testdata_tagged.txt')[0]
	graph.append(dic)
	dic = {}

	# Experiment 2
	for percent in param:
		dic[percent] = part_of_speech_tagging(training[percent][0], training[percent][1],
											training[percent][2], 3, False,
											'testdata_untagged.txt', 'testdata_tagged.txt')[0]
	graph.append(dic)
	dic = {}

	# Experiment 3
	for percent in param:
		dic[percent] = part_of_speech_tagging(training[percent][0], training[percent][1],
											training[percent][2], 2, True,
											'testdata_untagged.txt', 'testdata_tagged.txt')[0]
	graph.append(dic)
	dic = {}

	# Experiment 4
	for percent in param:
		dic[percent] = part_of_speech_tagging(training[percent][0], training[percent][1],
											training[percent][2], 3, True,
											'testdata_untagged.txt', 'testdata_tagged.txt')[0]
	graph.append(dic)

	# Graph the four curves
	# comp182.plot_lines(graph, "Accuracy of Part-of-speech Tagging with HMM", "Percentage of training data",
	# 				"Accuracy", ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"],
	# 				"D:\User\Documents\College\Spring 2017\COMP 182\HW6\Accuracy")

	return graph

# print run()
