import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scripts.utils import *
from models.simple_attention_model import SimpleAttentionModel
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser("NQA train parser")
parser.add_argument(
    "--model", default='SAM', help="Model"
)
parser.add_argument(
    "--num-epochs", type=int, default=10, help="Number of epochs"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate"
)
parser.add_argument(
    "--hidden-layer", type=int, default=100, help="Hidden layer dimension"
)
parser.add_argument(
    "--path", default='datasets/train_10k.json', help="Train data"
)
parser.add_argument(
    "--vectors-path", default='datasets/glove_vectors.pkl', help="Embedding data"
)
parser.add_argument(
    "--indices-path", default='datasets/glove_token_to_index.pkl', help="Indices data"
)
parser.add_argument(
    "--out-path", default='checkpoints/final_model.pth', help="Out path"
)
parser.add_argument(
    "--batch-size", type=int, default=32, help="Number of epochs"
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

    print("Loading data...")
    data = in_memory_data_loader(args.path)
    embedding_vector_file = open(args.vectors_path,"rb")
    embedding_vectors = pickle.load(embedding_vector_file)
    tokens_to_id_file = open(args.indices_path, "rb")
    tokens_to_index = pickle.load(tokens_to_id_file)
    print("Data loading finished...")

    number_of_iterations = len(data) // 32
    model = SimpleAttentionModel(300, args.hidden_layer, pretrained_embeddings = embedding_vectors).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    number_of_iterations = args.num_epochs * number_of_iterations
    for iteration_number in tqdm(range(number_of_iterations)):
        questions, texts, labels = get_mini_batch(data, tokens_to_index, device=device, batch_size=args.batch_size)
        out = model(questions, texts)
        loss = criterion(out, labels)
        if iteration_number % 100 == 0:
            print(loss.data)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()
        
    torch.save(model.state_dict(), args.out_path)