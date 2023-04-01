import torch
import torch.nn as nn
import numpy as np
from types import GeneratorType

from util.utils import load_data_set, gen_batch_sequence
from model.word_embedding import WordEmbedding
from model.aggregation_predictor import AggregationPredictor


class Seq2SQL(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_depth):
        super(Seq2SQL, self).__init__()

        self.gpu = torch.cuda.is_available()
        if (hidden_size & 1 != 0):
            raise ValueError(
                'hidden size must be even, since this is a bidirectional network')
        self.hidden_size = hidden_size

        self.word_emb = WordEmbedding(bert_model_name)
        self.word_emb_size = self.word_emb.bert_model.config.hidden_size
        self.aggregator = AggregationPredictor(input_layer=self.word_emb_size,
                                               hidden_size=hidden_size, num_layers=num_depth, gpu=True)
        self.CE = nn.CrossEntropyLoss()
        if (torch.cuda.is_available):
            self.to('cuda')

    def forward(self, queries, col):
        x_embed, x_lengths = self.word_emb.gen_x_batch(
            q_batch=queries, col_batch=col)

        agg_score = self.aggregator(x_embed.last_hidden_state, x_lengths)

        return (agg_score,)

    # TODO: clean this loss funtion
    def loss(self, score, truth_num):
        agg_score = score[0]
        loss = 0
        agg_truth = list(map(lambda x: x[0], truth_num))
        data = torch.from_numpy(np.array(agg_truth))
        if self.gpu:
            agg_truth_var = data.cuda()
        else:
            agg_truth_var = data

        loss += self.CE(agg_score, agg_truth_var.long())
        return loss

    def gen_query(self, score, query_batch, col_batch, raw_query, raw_col):
        agg_score = score[0]
        B = len(query_batch)
        agg_pred = np.argmax(agg_score.data.cpu().numpy(), axis=1)
        pred_queries = []
        for i in range(len(agg_pred)):
            pred_queries.append({'agg': agg_pred[i]})

        return pred_queries

    def check_accuracy(self, pred_queries, ground_truth_queries):
        tot_err = agg_err = 0

        for b, (pred_qry, ground_truth_qry) in enumerate(zip(pred_queries, ground_truth_queries)):
            good = True

            agg_pred = pred_qry['agg']
            agg_gt = ground_truth_qry['agg']
            if agg_pred != agg_gt:
                agg_err += 1
                good = False

            if good == False:
                tot_err += 1

        return np.array((agg_err)), tot_err
