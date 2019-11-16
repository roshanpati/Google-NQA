import json
import numpy as np

np.random.seed(42)

def in_memory_data_loader(filepath):
    """
    Returns a list where each entry is a dict with keys - 
        'document_text' - list of tokens in a particular long answer candidate
        'question' - list of tokens in the question
        'label' - 0/1
    
    Note: This currently only loads data corresponding to long answer

    TODO: Use pytorch dataloader instead?
    """
    data_final = []
    f = open(filepath)
    for line in f:
        entry = json.loads(line)
        document_text = entry['document_text'].lower().split(' ')
        data_entry_positive = {}
        data_entry_negative = {}
        data_entry_positive['question'] = data_entry_negative['question'] = entry['question_text'].lower().split()
        long_answer_start = entry['annotations'][0]['long_answer']['start_token']
        long_answer_end = entry['annotations'][0]['long_answer']['end_token']
        if long_answer_start != -1 and long_answer_end != -1:
            data_entry_positive['text'] = document_text[entry['annotations'][0]['long_answer']['start_token']:entry['annotations'][0]['long_answer']['end_token']]
            data_entry_positive['label'] = 1
            data_final.append(data_entry_positive)
        candidate = entry['long_answer_candidates'][np.random.randint(len(entry['long_answer_candidates']))]
        if candidate['start_token'] == long_answer_start and candidate['end_token'] == long_answer_end:
            continue
        data_entry_negative['text'] = document_text[candidate['start_token']:candidate['end_token']]
        data_entry_negative['label'] = 0
        data_final.append(data_entry_negative)
    return data_final

if __name__ == "__main__":
    data = in_memory_data_loader('../datasets/train_10k.json')
    print(len(data))