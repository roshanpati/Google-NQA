import json
import numpy as np
from nltk import word_tokenize

np.random.seed(42)

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
    while (len(sentence) <= final_length):
        sentence.append(pad_token)
    return sentence


def get_mini_batch(data, batch_size=32, pad_token='@@PAD@@'):
    """
    Padding questions and texts to have size 100
    """
    max_index = data.shape[0]
    indices = np.random.randint(0, max_index, size=batch_size)
    questions = [pad_sentence(data[i]['question'], pad_token, 100) for i in indices]
    texts = [pad_sentence(data[i]['text'], pad_token, 100) for i in indices]
    labels = [data[i]['label'] for i in indices]
    return questions, texts, labels

if __name__ == "__main__":
    data = in_memory_data_loader('../datasets/train_10k.json')
    mini_batch = get_mini_batch(data)
    assert mini_batch.shape[0] == 32