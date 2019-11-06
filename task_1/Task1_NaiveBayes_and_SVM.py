#!/usr/bin/env python
# coding: utf-8

# In[155]:


# IMPORTS
#import numpy as np
import glob
import math
import sys
import subprocess
import statistics
from nltk.stem import PorterStemmer
import operator as op
from functools import reduce
from decimal import Decimal
from scipy.stats import binom

# CONSTANTS
NUM_FOLDS = 10
DATA_SIZE = 2000
POS_PROB = 0.5
NEG_PROB = 1 - POS_PROB


# In[156]:


# READ IN FILES
def get_data_set(stemming, presence, bigrams, uni_and_bigrams, cutoff):
    """Returns a dict mapping fold number to data fold.
    
    Returns:
        fold number (int) -> list of documents 
        (list of list of strings)
    """
    pos_path = 'POS-tokenized/POS/*.tag'
    neg_path = 'NEG-tokenized/NEG/*.tag'
    pos_data = _read_data(pos_path)
    neg_data = _read_data(neg_path)

    # split data into 10 stratified folds
    data_set = {}
    for i in range(NUM_FOLDS):
        data_set[i] = []
    
    feature_count = 0
    for i, docs in enumerate(zip(pos_data, neg_data)):
        pos_doc, neg_doc = docs
        if stemming:
            porter_stemmer = PorterStemmer()
            pos_doc = _apply_stemming(porter_stemmer, pos_doc)
            neg_doc = _apply_stemming(porter_stemmer, neg_doc)        
        if bigrams:
            bi_pos_doc = _unigrams_to_bigrams(pos_doc)
            bi_neg_doc = _unigrams_to_bigrams(neg_doc)
            if uni_and_bigrams:
                pos_doc = pos_doc + bi_pos_doc
                neg_doc = neg_doc + bi_neg_doc
            else:
                pos_doc = bi_pos_doc
                neg_doc = bi_neg_doc
        if cutoff > 0:
            pos_doc = _apply_feature_cutoff(pos_doc, cutoff)
            neg_doc = _apply_feature_cutoff(neg_doc, cutoff)
        if presence:
            pos_doc = set(pos_doc)
            neg_doc = set(neg_doc)
            
        data_set[i%NUM_FOLDS].append((pos_doc, "POS"))
        data_set[i%NUM_FOLDS].append((neg_doc, "NEG"))
        
        feature_count += len(pos_doc) + len(pos_doc)
    print("feature_count = {}".format(feature_count))
    return data_set

def _read_data(folder_path):
    files = glob.glob(folder_path)
    data = []
    for file_name in files:
        with open(file_name) as fp:
            document = [word.strip("\n") for word in fp.readlines()]
            data.append(document)
    return data

def _apply_stemming(porter_stemmer, doc):
    return [porter_stemmer.stem(word) for word in doc]

def _unigrams_to_bigrams(doc):
    return [word1 + word2 
            for word1, word2 
            in zip(doc[:-1], doc[1:])]

def _apply_feature_cutoff(doc, cutoff):
    token_count = dict()
    for word in doc:
        if word not in token_count:
            token_count[word] = 0
        token_count[word] = token_count[word] + 1
    for word in doc:
        if token_count[word] < cutoff:
            doc.remove(word)
    return doc


# In[134]:


def cross_validate_data_set(data_set):
    """ Yields data set organised into training and test sets.
    
    Args:
        data_set: a dict mapping fold number to fold data.
        
    Returns: 
        A tuple (training_set, test_set). Each of these entries
        represents a list of docs (a list of list of strings).
    """
    for test_num in data_set.keys():
        training_set = []
        test_set = []
        for fold_num, curr_fold in data_set.items():
            if fold_num == test_num:
                test_set = curr_fold
            else:
                training_set.extend(curr_fold)
        yield (training_set, test_set)


# In[142]:


def classify_NB(training_set, test_set):
    """ Classifies docs in test_set based on training_set.
    
    Args:
        training_set: training docs (list of list of strings).
        test_set: testing docs (list of list of strings).
        
    Returns:
        A list of the outcomes for each doc in the test_set.
    """
    log_probs = calculate_log_probs(training_set)
    correct_classifications = []
    for test_doc in test_set:
        pos_sum = POS_PROB
        neg_sum = NEG_PROB
        test_doc_words, true_sentiment = test_doc
        for word in test_doc_words:
            if word not in log_probs:
                continue
            pos_log_prob, neg_log_prob = log_probs[word]
            pos_sum += pos_log_prob
            neg_sum += neg_log_prob
        result_sentiment = "POS"
        if pos_sum < neg_sum:
            result_sentiment = "NEG"    
        
        if result_sentiment == true_sentiment:
            correct_classifications.append(1)
        else:
            correct_classifications.append(0)
    return correct_classifications

def calculate_log_probs(training_set):
    """Returns logged probabilities of each word
    
    P(f_i|c) = count(f_i,c)/sum_i count(f_i,c)
    
    Args:
        training_set: documents to be trained on; list of 
            (list of words, sentiment (string)).
    
    Returns:
        A dictionary mapping words (string) to a 
        (pos log prob (float), neg log prob (float)) tuple.
    """
    # Find word_counts: word -> (pos_count, neg_count)
    word_counts = {}
    total_positive_words = 0
    total_negative_words = 0
    for doc_words, sentiment in training_set:
        for word in doc_words:
            if word not in word_counts:
                word_counts[word] = (1, 1) # laplace smoothing
                total_positive_words += 1
                total_negative_words += 1
            pos_count, neg_count = word_counts[word]
            if sentiment == "POS":
                word_counts[word] = (pos_count + 1, neg_count)
                total_positive_words += 1
            else:
                word_counts[word] = (pos_count, neg_count + 1)
                total_negative_words += 1
    
    # Log values and find total positive and negative counts.
    log_probs = {}
    for word, counts in word_counts.items():
        pos_counts, neg_counts = counts
        pos_log_prob = math.log(float(pos_counts)/total_positive_words)
        neg_log_prob = math.log(float(neg_counts)/total_negative_words)
        log_probs[word] = (pos_log_prob, neg_log_prob)
    return log_probs
            


# In[143]:


def run_naive_bayes(stemming, presence, bigrams, uni_and_bigrams, cutoff): 
    """Run naive bayes on data set and print results."""
    data_set = get_data_set(stemming, presence, bigrams, uni_and_bigrams, cutoff)
    organised_sets = cross_validate_data_set(data_set)
    correct_classifications = []
    accuracies = []
    for training_set, test_set in organised_sets:
        fold_classifications = classify_NB(training_set, test_set)
        correct_classifications.append(fold_classifications)
        accuracy = fold_classifications.count(1)/len(fold_classifications)
        accuracies.append(accuracy)
    return (accuracies, correct_classifications)


# In[137]:


# SVM_LIGHT
def write_data_set_to_files(data_set, fp, word_encodings):
    new_word_count = len(word_encodings) +1
    for doc_words, sentiment in data_set:
        target = "1" if sentiment == "POS" else "-1"
        word_counts = dict()
        for word in doc_words:
            if word not in word_encodings:
                word_encodings[word] = new_word_count
                new_word_count += 1
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] = word_counts[word] + 1
        feature_values = [(word_encodings[word], count)
                         for (word, count) in word_counts.items()]
        feature_values.sort()
        feature_values = ["{}:{}".format(encoding, count) 
                          for (encoding, count) in feature_values]
        feature_values = " ".join(feature_values)
        line = "{} {}\n".format(target, feature_values)
        fp.write(line)


# In[138]:


def run_svm(stemming, presence, bigrams, uni_and_bigrams, cutoff):
    data_set = get_data_set(stemming, presence, bigrams, uni_and_bigrams, cutoff)
    organised_sets = cross_validate_data_set(data_set)
    all_classifications = []
    accuracies = []
    for training_set, test_set in organised_sets:
        word_encodings = dict()
        # Train
        with open("training_set.txt", "w+") as training_fp:
            write_data_set_to_files (training_set, 
                                     training_fp,
                                     word_encodings)
        subprocess.call(["svm_light/svm_learn", 
                         "training_set.txt", 
                         "model_file.txt"])
        # Classify
        with open("test_set.txt", "w+") as test_fp:
            write_data_set_to_files (test_set, 
                                     test_fp,
                                     word_encodings)
        subprocess.call(["svm_light/svm_classify",
                         "test_set.txt",
                         "model_file.txt",
                         "predictions.txt"])
        # Test
        fold_classifications = []
        with open("predictions.txt") as pred_fp:
            for test_doc in test_set:
                result = float(pred_fp.readline())
                test_doc_words, true_sentiment = test_doc
                if result >= 0 and true_sentiment == "POS" or result < 0 and true_sentiment == "NEG":
                    fold_classifications.append(1)
                else:
                    fold_classifications.append(0)
        all_classifications.append(fold_classifications)
        accuracies.append(fold_classifications.count(1)/len(fold_classifications))
    return accuracies, all_classifications


# In[139]:


def _sign_test(base_classifications, new_classifications):
    """Runs sign test.
    
    Args:
        base_classifications: list of 1s or 0s. 1 if correctly classified and 0 otherwise.
        new_classifications: list of 1s or 0s. 1 if correctly classified and 0 otherwise.
    
    Returns:
        p-value.
    """
    better_count = 0
    worse_count = 0
    equal_count = 0
    for base_class, new_class in zip(base_classifications, new_classifications):
        if base_class == new_class:
            equal_count += 1
        elif new_class > base_class:
            better_count += 1
        else:
            worse_count += 1
    return _sign_test_aux(better_count, worse_count, equal_count)
    
def _sign_test_aux(better_count, worse_count, equal_count):
    N = 2*math.ceil(equal_count/2) + better_count + worse_count
    k = math.ceil(equal_count/2) + min(better_count, worse_count)
    return 2 * binom.cdf(k, N, 0.5)


def sign_test_on_multiple_folds(stemming=False, presence=False, bigrams=False, uni_and_bigrams=False, cutoff=0):    
    nb_accuracies, nb_classifications = run_naive_bayes(stemming, presence, bigrams, uni_and_bigrams, cutoff)
    svm_accuracies, svm_classifications = run_svm(stemming, presence, bigrams, uni_and_bigrams, cutoff)
    
    nb_average = sum(nb_accuracies) / len(nb_accuracies)
    svm_average = sum(svm_accuracies) / len(svm_accuracies)
    print("NB = {}\nAverage = {}\nSVM={}\nAverage = {}".format(nb_accuracies, nb_average, svm_accuracies, svm_average))
    
    return nb_classifications, svm_classifications


def _marked_list_to_p_value(class_1_marked, class_2_marked):
    c1_flattened = _flatten_list(class_1_marked)
    c2_flattened = _flatten_list(class_2_marked)
    p_value = float(_sign_test(c1_flattened, c2_flattened))
    print("p-value = {}".format(p_value))
    

def _flatten_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


# In[150]:


print("(1)")
nb_1, svm_1 = sign_test_on_multiple_folds(stemming=True, presence=False, bigrams=False, uni_and_bigrams=False, cutoff=0)
print("\n(2)")
nb_2, svm_2 = sign_test_on_multiple_folds(stemming=True, presence=True, bigrams=False, uni_and_bigrams=False, cutoff=0)
print("\n(3)")
nb_3, svm_3 = sign_test_on_multiple_folds(stemming=True, presence=True, bigrams=True, uni_and_bigrams=False, cutoff=0)
print("\n(4)")
nb_4, svm_4 = sign_test_on_multiple_folds(stemming=True, presence=True, bigrams=True, uni_and_bigrams=True, cutoff=0)


# In[151]:


print("nb_1")
_marked_list_to_p_value(nb_1, svm_1)
_marked_list_to_p_value(nb_1, svm_2)
_marked_list_to_p_value(nb_1, svm_3)
_marked_list_to_p_value(nb_1, svm_4)

print("\nnb_2")
_marked_list_to_p_value(nb_2, svm_1)
_marked_list_to_p_value(nb_2, svm_2)
_marked_list_to_p_value(nb_2, svm_3)
_marked_list_to_p_value(nb_2, svm_4)

print("\nnb_3")
_marked_list_to_p_value(nb_3, svm_1)
_marked_list_to_p_value(nb_3, svm_2)
_marked_list_to_p_value(nb_3, svm_3)
_marked_list_to_p_value(nb_3, svm_4)

print("\nnb_4")
_marked_list_to_p_value(nb_4, svm_1)
_marked_list_to_p_value(nb_4, svm_2)
_marked_list_to_p_value(nb_4, svm_3)
_marked_list_to_p_value(nb_4, svm_4)


# In[153]:


print("(1)")
nb_1, svm_1 = sign_test_on_multiple_folds(stemming=True, presence=False, bigrams=False, uni_and_bigrams=False, cutoff=2)
print("\n(2)")
nb_2, svm_2 = sign_test_on_multiple_folds(stemming=True, presence=True, bigrams=False, uni_and_bigrams=False, cutoff=2)
print("\n(3)")
nb_3, svm_3 = sign_test_on_multiple_folds(stemming=True, presence=True, bigrams=True, uni_and_bigrams=False, cutoff=2)
print("\n(4)")
nb_4, svm_4 = sign_test_on_multiple_folds(stemming=True, presence=True, bigrams=True, uni_and_bigrams=True, cutoff=2)


# In[154]:


print("nb_1")
_marked_list_to_p_value(nb_1, svm_1)
_marked_list_to_p_value(nb_1, svm_2)
_marked_list_to_p_value(nb_1, svm_3)
_marked_list_to_p_value(nb_1, svm_4)

print("\nnb_2")
_marked_list_to_p_value(nb_2, svm_1)
_marked_list_to_p_value(nb_2, svm_2)
_marked_list_to_p_value(nb_2, svm_3)
_marked_list_to_p_value(nb_2, svm_4)

print("\nnb_3")
_marked_list_to_p_value(nb_3, svm_1)
_marked_list_to_p_value(nb_3, svm_2)
_marked_list_to_p_value(nb_3, svm_3)
_marked_list_to_p_value(nb_3, svm_4)

print("\nnb_4")
_marked_list_to_p_value(nb_4, svm_1)
_marked_list_to_p_value(nb_4, svm_2)
_marked_list_to_p_value(nb_4, svm_3)
_marked_list_to_p_value(nb_4, svm_4)

