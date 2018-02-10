# Parts-of-Speech-Tagging
Part-of-speech tagging, or just tagging for short, is the process of assigning a part of speech or other syntactic class marker to each word in a corpus. Examples of tags include ‘adjective,’ ‘noun,’ ‘adverb,’ etc.
The input to a tagging algorithm is a string of words and a specified tagset. The output is a single best tag for each word.

This project utilized first and second order Hidden Markov Models on a training dataset with already tagged parts of speech to train the model and implemented Viterbi’s Algorithm in Python to tag parts of speech on a body of test text. 

The main code is in the file main.py, with helper module helper.py and provided.py. The "run" method in the main file will train/test the model using the data in the repo and plot out the accurary of our prediction.

This is a homework project in COMP 182
