import torch.nn as nn
import numpy as np
from util.utils import run_lstm, col_name_encode

class SelectionPredictor(nn.Module):
    def __init__(self,input_layer,hidden_size,num_layers,max_tok_num,gpu):
        super(SelectionPredictor, self).__init__()
        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.LSTM(input_size=input_layer, hidden_size=hidden_size // 2,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)
        
        self.sel_att = nn.Linear(in_features=hidden_size, out_features=1)
        self.sel_col_name_enc = nn.LSTM(input_size=input_layer, hidden_size=hidden_size // 2,
                                num_layers=num_layers, batch_first=True,
                                dropout=0.3, bidirectional=True)
        self.sel_out_K = nn.Linear(hidden_size, hidden_size)
        self.sel_out_col = nn.Linear(hidden_size, hidden_size)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, 1))
        self.softmax = nn.Softmax()
        
        if(gpu):
            self.sel_lstm = self.sel_lstm.to('cuda')
            self.sel_col_name_enc = self.sel_col_name_enc.to('cuda')
            self.sel_att  = self.sel_att.to('cuda')
            self.softmax = self.softmax.to('cuda')
            self.sel_out_K  = self.sel_out_K.to('cuda')
            self.sel_out_col  = self.sel_out_col.to('cuda')
            self.sel_out  = self.sel_out.to('cuda')
    
    def forward(self,x_input, x_len,col_inp_var, col_name_len, col_len,col_num):
        
        B = len(x_len)
        max_x_len = max(x_len)
        
        e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.sel_col_name_enc)
        h_enc, _ = run_lstm(self.sel_lstm, x_input, x_len) 
       
        att_val = self.sel_att(h_enc)  
        att_val = att_val.squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        
        att = self.softmax(att_val)
        K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        K_sel_expand = K_sel.unsqueeze(1)

        sel_score = self.sel_out(self.sel_out_K(K_sel_expand) + self.sel_out_col(e_col)).squeeze()
        max_col_num = max(col_num)
        
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx][num:] = -100

        return sel_score