import numpy as np
import argparse
import json

def precision_topk_threshold(results, k = 5,  threshold = 15.00):
    ''' Calculates  precision given a results list and k or the threshold score'''
    '''
    Args:
        results(List): List of dict with each dict should have at least the following attributes
            'example_id'(Double): Unique identifier
            'answer_index'(int): Gold Truth index
            'scores'(List): List of scores assigned to candidate long_answers
        k(int): top_k to pick


    Returns:
        precision_topk(float): Precision for top_k selection
        average_answer_length(float): Average candidate answers count over the samples
        len_max (int): Maximum candidate answers count over the samples
        len_min (int): Min candidate answers count over the samples
        all_scores(List): List of all scores to generate scores level stats
        precision_threshold(float): Precision for threshold based selection
        picked_above_threshold(float): Average number of items picked above threshold

    '''
    LEN_MAX = 0
    LEN_MIN = 100000

    len_max = LEN_MAX
    len_min = LEN_MIN
    p_k = 0 #Calculates for how many examples the precision is a hit(for top_k)
    p_threshold = 0#Calculates for how many examples the precision is a hit(for threshold based ranking)
    count = 0 #Calculates average number of picked long answers based on the threshold
    average_answer_length = 0
    all_scores = []

    for sample in results:
        if sample['answer_index'] in np.flip(np.argsort(np.array(sample['scores'])))[:k]:
            p_k += 1
        average_answer_length += len(sample['scores'])
        if len(sample['scores']) > len_max:
            len_max = len(sample['scores'])
        if len(sample['scores']) < len_min:
            len_min = len(sample['scores'])
        all_scores += sample['scores']
        if sample['answer_index'] in np.argwhere(np.array(sample['scores']) >= threshold ).flatten():
            p_threshold += 1
        count += len(np.argwhere(np.array(sample['scores']) >= threshold ).flatten())

    t = len(results)
    precision_topk = float(p_k/t)
    average_answer_length /= t
    precision_threshold = float(p_threshold/t)
    picked_above_threshold = float(count/t)

    return precision_topk,average_answer_length,len_max,len_min,all_scores,precision_threshold, picked_above_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the metrics given results')
    parser.add_argument('--filepath', dest='filepath')
    parser.add_argument('--top_k', dest='k')
    parser.add_argument('--threshold', dest='threshold')

    args = parser.parse_args()
    print("Arguments passed are")
    print(args)

    filepath = args.filepath
    k = int(args.k)
    t = float(args.threshold)

    with open(filepath,"r") as f:
        results = json.load(f)
    output = precision_topk_threshold(results,k,t)
    min,max,mean,median = np.min(np.array(output[4])),np.max(np.array(output[4])),np.mean(np.array(output[4])),np.median(np.array(output[4]))
    print("precision_top_{0} is {1}.".format(k,output[0]))
    print("precision for threshold {0} is {1}.".format(t,output[5]))
    print("Average Number of answers picked above threshold {0} is {1}.".format(t,output[6]))
    print("Average answer length over samples(size {0}) is {1}.".format(len(results),output[1]))
    print("Max answer length over samples(size {0}) is {1}.".format(len(results),output[2]))
    print("Min answer length over samples(size {0}) is {1}.".format(len(results),output[3]))
    print("Scores Stats: Min = {0}, Max = {1}, Mean = {2}, Median = {3}".format(min,max,mean,median))
