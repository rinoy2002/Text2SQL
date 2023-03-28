import torch
import torch.nn as nn
import numpy as np
from util.utils import run_lstm
from model.word_embedding import WordEmbedding


class AggregationPredictor(nn.Module):
    def __init__(self, input_layer, hidden_size, num_layers, gpu):
        super(AggregationPredictor, self).__init__()
        self.agg_lstm = nn.LSTM(input_size=input_layer, hidden_size=hidden_size // 2,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)

        self.agg_att = nn.Linear(in_features=hidden_size, out_features=1)
        self.soft_max = nn.Softmax(dim=1)
        self.agg_out = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=6)
        )

        if (gpu):
            self = self.to('cuda')
            # self.agg_lstm = self.agg_lstm.to('cuda')
            # self.agg_att  = self.agg_att.to('cuda')
            # self.soft_max = self.soft_max.to('cuda')
            # self.agg_out  = self.agg_out.to('cuda')

    def forward(self, x_input, x_len):
        B = len(x_len)
        max_len = max(x_len)
        # [B * longestinput * hidden_size]
        h_n, _ = run_lstm(self.agg_lstm, x_input, x_len)

        # calculate the scalar attention score. [scalar, since one value for each input word.]
        att_val = self.agg_att(h_n)  # [B * longest_input * 1]
        att_val = att_val.squeeze()  # [B* longest_input]
        # print(x_len)
        for index, l in enumerate(x_len):
            if (l < max_len):
                att_val[index][l:] = -100

        att_prob_dist = self.soft_max(att_val,)  # [B * longest_input]
        att_prob_dist = att_prob_dist.unsqueeze(2).expand_as(
            h_n)  # [B * longest_input * hidden_size]
        K_agg = (h_n * att_prob_dist).sum(1)  # [B * longest_input]
        agg_score = self.agg_out(K_agg)  # [B * 6]
        return agg_score
