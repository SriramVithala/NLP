"""
author-gh: @adithya8
editor-gh: ykl7
"""

import os
import pickle
import numpy as np
import argparse

import torch

np.random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./baseline_models', help='Base directory of folder where models are saved')
    parser.add_argument('--input_filepath', type=str, default='./data/word_analogy_dev.txt', help='Word analogy file to evaluate on')
    parser.add_argument('--output_filepath', type=str, required=True, help='Predictions filepath')
    parser.add_argument("--loss_model", help="The loss function for training the word vector", default="nll", choices=["nll", "neg"])
    args, _ = parser.parse_known_args()
    return args

def read_data(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()
    
    candidate, test = [], []
    for line in data:
        a, b = line.strip().split("||")
        a = [i[1:-1].split(":") for i in a.split(",")]
        b = [i[1:-1].split(":") for i in b.split(",")]
        candidate.append(a)
        test.append(b)
    
    return candidate, test

def get_embeddings(examples, embeddings):

    """
    For the word pairs in the 'examples' array, fetch embeddings and return.
    You can access your trained model via dictionary and embeddings.
    dictionary[word] will give you word_id
    and embeddings[word_id] will return the embedding for that word.

    word_id = dictionary[word]
    v1 = embeddings[word_id]

    or simply

    v1 = embeddings[dictionary[word_id]]
    """

    norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
    normalized_embeddings = embeddings/norm

    embs = []
    for line in examples:
        temp = []
        for pairs in line:
            temp.append([ normalized_embeddings[dictionary[pairs[0]]], normalized_embeddings[dictionary[pairs[1]]] ])
        embs.append(temp)

    result = np.array(embs)
    
    return result

def evaluate_pairs(candidate_embs, test_embs):

    """
    Write code to evaluate a relation between pairs of words.
    Find the best and worst pairs and return that.
    """
    best_pairs = []
    worst_pairs = []

    ### TODO(students): start
    for i, candiate_embed in enumerate(candidate_embs):
        best_distance = 1000000
        worst_distance = 0
        best_pair = []
        worst_pair = []
        # spatial.distance.cosine(directionVec[j,:].reshape(-1,1), wordVec[j,i,:].reshape(-1,1)))
        pair_avg = np.zeros(len(candiate_embed[0][0]))
        for each_pair in candiate_embed:
            pair_diff = np.subtract(each_pair[0], each_pair[1])
            pair_avg = np.add(pair_avg, pair_diff)

        pair_avg = pair_avg / len(candiate_embed)
        test_pairs_list = test_embs[i]
        for index, test_pair in enumerate(test_pairs_list):
            test_pair_diff = np.subtract(test_pair[0], test_pair[1])
            distance = np.linalg.norm(pair_avg - test_pair_diff)

            if distance < best_distance:
                best_distance = distance
                best_pair = index
            
            if distance > worst_distance:
                worst_distance = distance
                worst_pair = index
        #         result = []
        # for i in range(4):
        #     temp = []
        #     for j in range(len(directionVec)):
        best_pairs.append(best_pair)
        worst_pairs.append(worst_pair)

    ### TODO(students): end
    
    return best_pairs, worst_pairs

def write_solution(best_pairs, worst_pairs, test, path):

    """
    Write best and worst pairs to a file, that can be evaluated by evaluate_word_analogy.pl
    """
    
    ans = []
    for i, line in enumerate(test):
        temp = [f'"{pairs[0]}:{pairs[1]}"' for pairs in line]
        temp.append(f'"{line[worst_pairs[i]][0]}:{line[worst_pairs[i]][1]}"')
        temp.append(f'"{line[best_pairs[i]][0]}:{line[best_pairs[i]][1]}"')
        ans.append(" ".join(temp))

    with open(path, 'w') as f:
        f.write("\n".join(ans))


if __name__ == '__main__':

    args = parse_args()

    loss_model = args.loss_model
    model_path = args.model_path
    input_filepath = args.input_filepath

    print(f'Model file: {model_path}/word2vec_{loss_model}.model')
    model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

    dictionary, embeddings = pickle.load(open(model_filepath, 'rb'))

    candidate, test = read_data(input_filepath)

    candidate_embs = get_embeddings(candidate, embeddings)
    test_embs = get_embeddings(test, embeddings)

    best_pairs, worst_pairs = evaluate_pairs(candidate_embs, test_embs)

    out_filepath = args.output_filepath
    print(f'Output file: {out_filepath}')
    write_solution(best_pairs, worst_pairs, test, out_filepath)