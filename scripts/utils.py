import json
import numpy as np
from nltk import word_tokenize
import torch

def complete_in_memory_data_loader(filepath='../datasets/train_1k.json'):
    """
    Returns a list where each entry is a dict with keys - 
        'question' - list of tokens in the question
        'long_answer_candidates' - list of long answer candidates
        'labels' - List of 0/1 corresponding to weather a candidate being the actual long answer
    
    Note: This currently only loads data corresponding to long answer

    """
    data_final = []
    f = open(filepath)
    for line in f:
        entry = json.loads(line)
        document_text = entry['document_text'].lower().split(" ")
        long_answer_candidates = []
        labels = []
        data_entry = {}
        data_entry['question'] = word_tokenize(entry['question_text'].lower())
        long_answer_start = entry['annotations'][0]['long_answer']['start_token']
        long_answer_end = entry['annotations'][0]['long_answer']['end_token']
        candidates = entry['long_answer_candidates']
        for candidate in candidates:
            label_here = 0
            if candidate['start_token'] == long_answer_start and candidate['end_token'] == long_answer_end:
                label_here = 1
            long_answer_candidate = document_text[candidate['start_token']:candidate['end_token']]
            long_answer_candidates.append(long_answer_candidate)
            labels.append(label_here)
        data_entry['long_answer_candidates'] = long_answer_candidates
        data_entry['labels'] = labels
        data_final.append(data_entry)
    return np.array(data_final)

def in_memory_data_loader(filepath='../datasets/train_10k.json'):
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
        document_text = entry['document_text'].lower().split(" ")
        data_entry_positive = {}
        data_entry_negative = {}
        data_entry_positive['question'] = data_entry_negative['question'] = word_tokenize(entry['question_text'].lower())
        long_answer_start = entry['annotations'][0]['long_answer']['start_token']
        long_answer_end = entry['annotations'][0]['long_answer']['end_token']
        if long_answer_start != -1 and long_answer_end != -1:
            data_entry_positive['text'] = document_text[entry['annotations'][0]['long_answer']['start_token']:entry['annotations'][0]['long_answer']['end_token']]
            # data_entry_positive['text'] = word_tokenize(' '.join(data_entry_positive['text']))
            data_entry_positive['label'] = 1
            data_final.append(data_entry_positive)
        candidate = entry['long_answer_candidates'][np.random.randint(len(entry['long_answer_candidates']))]
        if candidate['start_token'] == long_answer_start and candidate['end_token'] == long_answer_end:
            continue
        data_entry_negative['text'] = document_text[candidate['start_token']:candidate['end_token']]
        # data_entry_negative['text'] = word_tokenize(' '.join(data_entry_negative['text']))
        data_entry_negative['label'] = 0
        data_final.append(data_entry_negative)
    return np.array(data_final)

def pad_sentence(sentence, pad_token, final_length):
    sentence = sentence[:final_length]
    while (len(sentence) < final_length):
        sentence.append(pad_token)
    return sentence

def sentence_to_indices(sentence, tokens_to_index, unknown_token='@@UNK@@'):
    index_of_unk = tokens_to_index[unknown_token]
    return [tokens_to_index.get(word, index_of_unk) for word in sentence]

def get_mini_batch(data, tokens_to_index, batch_size=32, pad=True, pad_token='@@PAD@@', unknown_token='@@UNK@@', device=torch.device('cpu')):
    """
    Padding questions and texts to have size 100
    Returns torch tensor with indices
    """
    max_index = data.shape[0]
    indices = np.random.randint(0, max_index, size=batch_size)
    questions = []
    texts = []
    if pad == True:
        questions = [sentence_to_indices(pad_sentence(data[i]['question'], pad_token, 100), tokens_to_index) for i in indices]
        texts = [sentence_to_indices(pad_sentence(data[i]['text'], pad_token, 100), tokens_to_index) for i in indices]
    else:
        questions = [sentence_to_indices(data[i]['question'], tokens_to_index) for i in indices]
        texts = [sentence_to_indices(data[i]['text'], tokens_to_index) for i in indices]
    questions = torch.Tensor(questions).long().to(device)
    texts = torch.Tensor(texts).long().to(device)
    labels = torch.Tensor([data[i]['label'] for i in indices]).unsqueeze(1).float().to(device)
    return questions, texts, labels


def get_contiguous_mini_batch_for_test(data, tokens_to_index, start_index, end_index, batch_size=32, pad=True, pad_token='@@PAD@@', unknown_token='@@UNK@@', device=torch.device('cpu')):
    """
    Padding questions and texts to have size 100
    Returns torch tensor with indices

    This should return a question, along with all the long answer candidates
    """
    max_index = data.shape[0]
    indices = range(start_index, end_index)
    questions = []
    texts = []
    if pad == True:
        questions = [sentence_to_indices(pad_sentence(data[i]['question'], pad_token, 100), tokens_to_index) for i in indices]
        texts = [sentence_to_indices(pad_sentence(data[i]['text'], pad_token, 100), tokens_to_index) for i in indices]
    else:
        questions = [sentence_to_indices(data[i]['question'], tokens_to_index) for i in indices]
        texts = [sentence_to_indices(data[i]['text'], tokens_to_index) for i in indices]
    questions = torch.Tensor(questions).long().to(device)
    texts = torch.Tensor(texts).long().to(device)
    labels = torch.Tensor([data[i]['label'] for i in indices]).unsqueeze(1).float().to(device)
    return questions, texts, labels

if __name__ == "__main__":
    data = in_memory_data_loader('../datasets/train_10k.json')
    mini_batch = get_mini_batch(data)
    assert mini_batch.shape[0] == 32
