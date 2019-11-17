import json
import argparse
import random

class DataLoader():
    def __init__(self,filepath):
        self.original_filepath = filepath
        self.orginal_file = None #In the original file each sample is a
        self.dataset_size = 0
        self.train_size = 0
        self.test_size = 0
    def split_train_test(self,split):
        '''Reads the data line by line, converts into a json list and splits the data into train and test
         according to the split value and stores then in then same directory with suffix _train_{train_size}
         or _test_{test_size} added'''
        self.split = split
        with open(self.original_filepath) as json_file:
            content = json_file.readlines()
        self.dataset_size = len(content)
        nqa = [json.loads(line) for line in content]
        random.shuffle(nqa)
        train_nqa = nqa[:int(self.dataset_size*self.split)]
        test_nqa = nqa[int(self.dataset_size*self.split):]
        self.train_size = len(train_nqa)
        self.test_size = len(test_nqa)
        self.train_path = '/'.join(self.original_filepath.split('/')[:-1])+"/"+self.original_filepath.split('/')[-1].split('.')[0]+"_train_{}.".format(self.train_size)+ \
                         self.original_filepath.split('/')[-1].split('.')[1]
        self.test_path = '/'.join(self.original_filepath.split('/')[:-1])+"/"+self.original_filepath.split('/')[-1].split('.')[0]+"_test_{}.".format(self.test_size)+ \
                         self.original_filepath.split('/')[-1].split('.')[1]

        with open(self.train_path,"w") as f:
            json.dump(train_nqa, f)
        with open(self.test_path,"w") as f:
            json.dump(test_nqa, f)

    def process_data(self,without_tags = True):
        self.read_and_make_single_json()
        # self.dataset_size =30000
        # self.modified_fie_path = self.modified_fie_path ='/'.join(self.original_filepath.split('/')[:-1])+"/nqa_modified_file_{}.json".format(self.dataset_size)
        with open(self.modified_fie_path,"r") as f:
            data = json.load(f)
        print("raw data loaded...........")
        qa_data = []
        # print("Chunk Number Processing is {}".format(k))
        for i in range(len(data)):
            temp = {}
            temp['question'] = data[i]['question_text']
            temp['doc_text'] = data[i]['document_text']
            temp['id'] = data[i]['example_id']
            temp['doc_url'] = data[i]['document_url']
            temp['split_text'] = temp['doc_text'].split(' ')

            la_start = data[i]['annotations'][0]['long_answer']['start_token']
            la_end = data[i]['annotations'][0]['long_answer']['end_token']
            if without_tags:
                temp['long_answer'] = ' '.join([item for item in temp['split_text'][la_start:la_end] if not ('<' in item or '>' in item) ])
            else:
                temp['long_answer'] = ' '.join([item for item in temp['split_text'][la_start:la_end]])

            long_answer_candidate_texts = []
            for long_an_can in data[i]['long_answer_candidates']:
                lac_start = long_an_can['start_token']
                lac_end = long_an_can['end_token']
                if lac_start == la_start and lac_end == la_end:
                    temp['long_answer_index'] = data[i]['long_answer_candidates'].index(long_an_can)
                if without_tags:
                    long_answer_candidate_text = ' '.join([item for item in temp['split_text'][lac_start:lac_end] if not ('<' in item or '>' in item) ])
                else:
                    long_answer_candidate_text = ' '.join([item for item in temp['split_text'][lac_start:lac_end]])
                long_answer_candidate_texts.append(long_answer_candidate_text)


            temp['long_answer_candidate_texts'] = long_answer_candidate_texts
            temp['q_a'] = temp['question']+" ? "+temp['long_answer']
            qa_data.append(temp)

        random.shuffle(qa_data)
        train_qla = data[:int(self.dataset_size*4/5)]
        test_qla = data[int(self.dataset_size*4/5):]
        return train_qla,test_qla

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filepath', dest='filepath')
    parser.add_argument('--split', dest='split')

    args = parser.parse_args()
    print("Arguments passed are")
    print(args)
    filepath = args.filepath
    split = float(args.split)

    # filepath = "../datasets/sampled_simplified_nq_train.json"
    dl = DataLoader(filepath)
    dl.split_train_test(split)
