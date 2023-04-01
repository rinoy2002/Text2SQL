import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np

from util.utils import gen_batch_sequence


class WordEmbedding:
    def __init__(self, bert_encoder, max_length=None, separator='[SEP]'):
        # super(WordEmbedding, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_encoder)
        self.bert_model = BertModel.from_pretrained(bert_encoder)
        self.max_length = max_length
        self.gpu = torch.cuda.is_available()
        self.separator = separator

        self.bert_args = {'add_special_tokens': True,
                          'return_token_type_ids': True,
                          'padding': 'longest',
                          'return_attention_mask': True,
                          'return_tensors': 'pt'}

        if (self.max_length != None):
            self.bert_args['max_length'] = + self.max_length
            self.bert_args['padding'] = 'max_length'

        for param in self.bert_model.parameters():
            param.requires_grad = False
        assert list(self.bert_model.parameters())[0].requires_grad == False

        if self.gpu:
            self.bert_model = self.bert_model.to('cuda')

    def gen_x_batch(self, q_batch, col_batch):
        '''
        Input: q_batch: list of tokenized query string i.e. List[List].
               col_batch: list of tokenzed header of the corresponding table header. List[List[List]]
        Output: bert_op : containing the last_hidden_layer and pooler_output
        '''
        batch_queries = [' '.join(x) for x in q_batch]
        if batch_queries == []:
            print(q_batch, col_batch)

        header_batch_list = list(
            map(lambda col: [x for tok in col for x in tok +
                [self.separator]], col_batch)
        )

        input_string_list = []
        input_string_lengths = []
        for i in range(len(header_batch_list)):
            merged_list = q_batch[i] + \
                [self.separator] + header_batch_list[i][:-1]
            input_string_lengths.append(len(merged_list))
            input_string_list.append(' '.join(merged_list))

        inp_encode = self.bert_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=input_string_list, **self.bert_args)
        if self.gpu:
            for key in inp_encode.keys():
                inp_encode[key] = inp_encode[key].to('cuda')
        bert_op = self.bert_model(**inp_encode)

        # odict_keys(['last_hidden_state', 'pooler_output'])
        return bert_op, input_string_lengths

    def gen_col_batch(self, col_batch):
        '''
        Input: col_batch: list of columns in a batch [[[str]]]
        Output: embedding : embedding of word [No. of column in a batch X (max column length in the batch + 2) X 768]
                embedding_len : word length 
                col_len : No. of column in a table

        '''
        col_len = []
        embedding = []
        embedding_len = []
        # zero array for appending
        zeros = np.zeros(768, dtype=np.float32)
        for i in range(len(col_batch)):
            col_len.append(len(col_batch[i]))
            for y in col_batch[i]:
                inp_encode = self.bert_tokenizer.encode_plus(
                    text=y, **self.bert_args)
                if self.gpu:
                    for key in inp_encode.keys():
                        inp_encode[key] = inp_encode[key].to('cuda')
                bert_encode = self.bert_model(**inp_encode)
                if bert_encode.last_hidden_state.is_cuda:
                    bert_encoding_numpy = torch.squeeze(
                        bert_encode.last_hidden_state, dim=0).cpu().detach().numpy()
                bert_encoding_numpy = torch.squeeze(
                    bert_encode.last_hidden_state, dim=0).cpu().numpy()
                embedding_len.append(bert_encoding_numpy.shape[0])
                embedding.append(bert_encoding_numpy)

        max_len = max(embedding_len)
        for i in range(len(embedding)):
            zeros_append = np.tile(zeros, (max_len-embedding_len[i], 1))
            embedding[i] = np.concatenate((embedding[i], zeros_append), axis=0)

        embedding = torch.tensor(embedding)
        embedding_len = np.array(embedding_len)
        if self.gpu:
            embedding = embedding.to('cuda')

        return embedding, embedding_len, col_len


def test_wordembed_module(train_sql, train_table, batch_size=32):
    word_emb = WordEmbedding('bert-base-uncased')
    start = 0
    end = batch_size
    num_x = len(train_sql)
    idxes = np.random.permutation(num_x)
    i = 0
    while start < num_x:
        end = start+batch_size if start+batch_size <= num_x else num_x
        ret_tuple = gen_batch_sequence(
            train_sql, train_table, idxes, start, end)
        if ret_tuple[0] == []:
            print(start, end)
        bert_op, input_lengths = word_emb.gen_x_batch(
            ret_tuple[0], ret_tuple[1])
        last_hidden_state, pooler_output = bert_op.last_hidden_state, bert_op.pooler_output

        assert last_hidden_state.size()[2] == 768
        assert last_hidden_state.size()[0] <= batch_size
        assert pooler_output.size()[1] == 768
        assert pooler_output.size()[0] <= batch_size

        if (i % 10 == 0):
            print(f"\ni={i}\nSanity Check!!\n")
            print(f"Start: {start}\nEnd: {end}\n")
            print(f"Size of LastHidden Layer Size:{last_hidden_state.size()}\nPooler output size:{pooler_output.size()}\n\
            LastHidden Layer: {last_hidden_state}\nPooler output: {pooler_output}\n")

        i = i+1
        start = end
