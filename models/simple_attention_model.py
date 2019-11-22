import torch
import torch.nn as nn
import pickle

class SimpleAttentionModel(nn.Module):
    """
    Represent both question and text in vector form and score = q_question' * W * q_text
    The score value is between 0 and 1 and can be trained using BCELoss

    """
    def __init__(self, embed_size, hidden_size, pretrained_embeddings = None, vocab_size = 400000):
        super(SimpleAttentionModel, self).__init__()
        self.model_embeddings = None
        if pretrained_embeddings is not None:
            self.model_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.model_embeddings = nn.Embedding(vocab_size, embed_size)
        self.questions_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.text_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear_layer = nn.Linear(2*hidden_size, 2*hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, questions, texts, sigmoid=False):
        """
        questions, texts - b * length * embed_size
        """
        batch_size = questions.shape[0]
        _, (h_questions, _) = self.questions_lstm(self.model_embeddings(questions)) # h shape = 2 * batch_size * hidden_size
        _, (h_text, _) = self.text_lstm(self.model_embeddings(texts))
        h_questions = h_questions.transpose(0,1).reshape((batch_size, -1)) # batch_size, 2 * hidden_size
        h_text = h_text.transpose(0,1).reshape((batch_size, -1))
        output = self.linear_layer(h_text).unsqueeze(1) # batch_size, 1, 2 * hidden_size
        h_questions = h_questions.unsqueeze(2) # batch_size, 2*hidden_size, 1
        output = torch.bmm(output, h_questions).squeeze(2) # batch_size, 1
        if sigmoid == True:
            output = self.sigmoid(output)
        return output