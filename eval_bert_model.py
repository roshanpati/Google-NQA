import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from models.BertAnswerClassification import BertAnswerClassification
from scripts.utils_squad_evaluate import *
from scripts.utils_squad import *
from transformers import *
from torch.utils.data import *
from scripts.utils import *

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser("NQA parser")
parser.add_argument(
    "--model", default=None, help="Model"
)
parser.add_argument(
    "--file-path", default='data/train_10k.json', help="Test/Dev data"
)
parser.add_argument(
    "--out-path", default='results/', help="Out path"
)
parser.add_argument(
    "--batch-size", type=int, default=3, help="Number of epochs"
)
parser.add_argument(
    "--distil", type=bool, default=False, help="Use distil bert"
)
parser.add_argument(
    "--model-path", default=None, help="Path to saved model state"
)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    if args.distil == True:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if args.model == 'BertAnswerClassification': # Classification case
        model = BertAnswerClassification(distil = False) # Distil version not available yet
    elif args.model == 'BertQABase': # Question answering case
        if args.distil == True:
            model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        else:
            model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    else:
        assert False, 'Unknown model'

    model.load_state_dict(torch.load(args.start_path))
    model.eval()
    print('Loading dataset')
    dataset, examples, features = load_and_cache_examples(args.file_path, args.distil, tokenizer, evaluate=True)
    print('Finished loading dataset')

    sampler = SequentialSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    results = []
    iterator = tqdm(iter(train_dataloader))
    for batch_num, batch in tqdm(enumerate(iterator)):
        with torch.no_grad():
            output = model(batch[0], attention_mask=batch[1])

        example_indices = batch[3]
        for i, example_index in enumerate(example_indices):
            unique_index = int(features[example_index.item()].unique_id)
            result_here = RawResult(unique_id=unique_index, start_logits=outputs[0][i], end_logits=outputs[1][i])
            results.append(result_here)
    
    prediction_file = os.path.join(args.out_path, "predictions.json")
    nbest_file = os.path.join(args.out_path, "nbest_predictions.json")
    null_log_odds_file = os.path.join(args.output_path, "null_odds.json")
    
    write_predictions(examples, features, results, n_best_size=20,
                    max_answer_length=30, do_lower_case=True, prediction_file,
                    nbest_file, null_log_odds_file, verbose_logging=False,
                    version_2_with_negative=True, null_score_diff_threshold=0)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    return results



