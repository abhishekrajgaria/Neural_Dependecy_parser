import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


class MultiClassClassifier(nn.Module):
    def __init__(
        self,
        word_emb_dim,
        take_mean: bool,
        with_label: bool,
        pos_emb_dim=50,
        label_emb_dim=50,
        hidden_dim=200,
        window_size=2,
    ):
        super(MultiClassClassifier, self).__init__()
        self.take_mean = take_mean
        self.with_label = with_label

        self.word_emb_dim = (
            word_emb_dim if (take_mean) else 2 * window_size * word_emb_dim
        )
        if not take_mean and with_label:
            self.word_emb_dim = 2 * self.word_emb_dim

        self.pos_emb_dim = pos_emb_dim if (take_mean) else 2 * window_size * pos_emb_dim
        self.label_emb_dim = (
            label_emb_dim if (take_mean) else 2 * window_size * label_emb_dim
        )

        self.pos_projection_layer = nn.Embedding(18, 50)
        self.label_projection_layer = nn.Embedding(75, 50)

        self.word_linear_layer = nn.Linear(self.word_emb_dim, hidden_dim, bias=True)
        self.pos_linear_layer = nn.Linear(self.pos_emb_dim, hidden_dim, bias=True)
        self.label_linear_layer = nn.Linear(self.label_emb_dim, hidden_dim, bias=True)

        self.final_linear_layer = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden_dim, 75, bias=True)
        )
        # self.final_linear_layer = nn.Linear(hidden_dim, 75, bias=True)

    def forward(self, word_rep_input, pos_rep_input, label_rep_input=None):
        # print("word_rep_input_shape ", word_rep_input.shape)

        word_rep_input = (
            word_rep_input.mean(dim=1).squeeze(dim=1)
            if (self.take_mean)
            else word_rep_input.view(-1, self.word_emb_dim)
        )
        # print("word_rep_input_shape ", word_rep_input.shape)

        pos_rep_input = self.pos_projection_layer(pos_rep_input)
        # print("pos_rep_input_shape ", pos_rep_input.shape)

        pos_rep_input = (
            pos_rep_input.mean(dim=1)
            if (self.take_mean)
            else pos_rep_input.view(-1, self.pos_emb_dim)
        )

        # print("pos_rep_input_shape ", pos_rep_input.shape)

        word_hidden_rep = self.word_linear_layer(word_rep_input)
        pos_hidden_rep = self.pos_linear_layer(pos_rep_input)

        label_hidden_rep = None

        if self.with_label:
            label_rep_input = self.label_projection_layer(label_rep_input)
            label_rep_input = (
                label_rep_input.mean(dim=1).squeeze()
                if (self.take_mean)
                else label_rep_input.view(-1, self.label_emb_dim)
            )
            label_hidden_rep = self.label_linear_layer(label_rep_input)

        hidden_rep = (
            (word_hidden_rep + pos_hidden_rep)
            if (not self.with_label)
            else (word_hidden_rep + pos_hidden_rep + label_hidden_rep)
        )

        # hidden_rep = nn.ReLU()(hidden_rep)

        output = self.final_linear_layer(hidden_rep)

        return output
