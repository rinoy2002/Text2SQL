import json
from types import GeneratorType
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data_set(dataset_name):
    '''
    Input: 
        dataset_name - str {train, test, dev}
    Output: 
        (sql_data, table_data) - (list, dictionary): containg the tokenized queries and the table dataset.

    Universal function to load the dataset
    '''
    print(f"Loading {dataset_name} dataset")
    sql_file = './data/tokenized_'+dataset_name+'.jsonl'
    tables_file = './data/tokenized_'+dataset_name+'.tables.jsonl'
    sql_query_count = 0
    table_count = 0
    sql_data = []
    table_data = {}

    with open(sql_file) as lines:
        for line in lines:
            # remove white space before and after
            sql = json.loads(line.strip())
            sql_data.append(sql)
            sql_query_count += 1
    with open(tables_file) as lines:
        for line in lines:
            tab = json.loads(line.strip())
            table_data[tab['id']] = tab
            table_count += 1
    print(f"Loaded {sql_query_count} queries and {table_count} tables")
    return sql_data, table_data


def gen_batch_sequence(sql_data, table_data, idxes, start, end):
    question_seq = []
    column_seq = []
    number_of_col = []
    answer_seq = []
    query_seq = []
    ground_truth_cond_seq = []
    raw_data = []
    for i in range(start, end):
        sql = sql_data[idxes[i]]
        table_id = sql['table_id']
        question_seq.append(sql['tokenized_question'])
        column_seq.append(table_data[table_id]['tokenized_header'])
        number_of_col.append(len(table_data[table_id]['header']))
        answer_seq.append((sql['sql']['agg'],
                           sql['sql']['sel'],
                           len(sql['sql']['conds']),
                           tuple(x[0] for x in sql['sql']['conds']),
                           tuple(x[1] for x in sql['sql']['conds'])
                           ))
        query_seq.append(sql['tokenized_query'])
        ground_truth_cond_seq.append(sql['sql']['conds'])
        raw_data.append(
            (sql['question'], table_data[table_id]['header'], sql['query']))

    return (question_seq, column_seq, number_of_col, answer_seq, query_seq, ground_truth_cond_seq, raw_data)


#  in the code for the reference paper they sorted the inputs (acc. to the size)
# TODO: Check if inp_length is still required or not.
def run_lstm(lstm, inp, inp_length, prev_hidden=None):
    '''
    Input: 
        lstm (nn.LSTM)      - the name of the lstm variable that needs to be run
        inp (a tensor)      - the input in the for [Batch size , num_tok, last_layer]
        inp_length (list)   - an array that contains the length of each element in the batch size = batch size
        pre_hidden (tensor) - hidden layer values of the previous lstm layer
    Output: 
        Same as nn.LSTM

    This function is used to run the LSTM Layer
    '''
    if prev_hidden != None:
        ret_h, ret_c = lstm(inp, prev_hidden)
    else:
        ret_h, ret_c = lstm(inp)
    return ret_h, ret_c


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    # Encode the columns.
    # The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len - 1]
    ret = torch.FloatTensor(len(col_len), max(col_len),
                            name_out.size()[1]).zero_()
    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st + cur_len]
        st += cur_len
    ret_var = torch.tensor(ret).to('cuda')
    return ret_var, col_len


def generate_batch_query(sql_data, idx, start, end):
    # TODO: this fucntion is redundant please find a proper alternative by gen_batch_sequence.
    '''
    Input: 
        sql_data (List)   -  a list of sql query dictionary [this is the original data and not shuffled]
        idx  (List)       - the order in which the dataset is shuffled
        start (int)       - start index
        end (int)         - end index

    Output:
        query_gt (list)   - containing the ground truth sql query of the batch
        table_id (list)   - list containing the tables sql queries.
    '''
    query_gt = []
    table_id = []
    for i in range(start, end):
        query_gt.append(sql_data[idx[i]]['sql'])
        table_id.append(sql_data[idx[i]]['table_id'])
    return query_gt, table_id


def plot_curve(x_item, y_item, item_name, dataLength, format='png'):
    '''
    Input: 
        x_item (list, int, or generator): this takes in 
                                                    1. Max value from zero in the x axis (int)
                                                    2. The range of values in the x axis (generator)
                                                    3. The values in the x axis (list)
        y_item (list): this takes in the values f(x) to be ploted
        item_name: the name to be plotted on the x axis
        dataLength: the size of the dataset trained on
        format: the format in which the image is to be saved.

    Output: 
        None.

    Doc: 
        Generate plost for the accuray and loss function curves and save it according to the current
        date and time in the folder named './Graphs'
    '''
    if isinstance(x_item, GeneratorType):
        x_item = list(x_item)
    if isinstance(x_item, int):
        x_item = list(range(x_item))
    plt.plot(x_item, y_item)

    plt.xlabel("EPOCHS")
    plt.ylabel(item_name)

    # plt.show()
    time = datetime.now()
    day_time_str = time.strftime("%H%M%S-%d%m%Y")

    plt.savefig(
        f'./Graphs/{day_time_str}_{len(y_item)}EP_{dataLength}trainingqueries_{item_name}.{format}', dpi=300, format=format)
    plt.show()
    return


def create_toy_dataset(actual_queries, table_data, num_samples):
    '''
    Input: 
        actual_queries - list of natural language queries
        table_data     - Contains the 
    Output:

    Doc:
        Based on the num_samples we create the required mini dataset.
    '''

    idx_list = np.random.permutation(len(actual_queries))[:num_samples]
    toy_queries = list(actual_queries[x] for x in idx_list)
    toy_tables = {}
    for x in range(num_samples):
        toy_tables[toy_queries[x]['table_id']
                   ] = table_data[toy_queries[x]['table_id']]
    return toy_queries, toy_tables


def epoch_train(model, optimizer, batch_size, sql_queries, table_data):
    '''
    Input: 
        model: a pytorch model subclass of nn.Module
        optimizer: a pytorch optimizer 
        batch_size: batch size of the 
    Output:
        loss: total cumulative loss of all 3 modules
    '''
    model.train()
    num_queries = len(sql_queries)
    perm = np.random.permutation(num_queries)
    cumulative_loss = 0.0
    start = 0

    while start < num_queries:
        end = start + batch_size if start + \
            batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, ground_truth_cond_seq, raw_data = \
            gen_batch_sequence(sql_queries, table_data, perm, start, end)

        score = model.forward(q_seq, col_seq)
        loss = model.loss(score, ans_seq)
        cumulative_loss += loss.data.cpu().numpy() * (end - start)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        start = end
    return cumulative_loss / len(sql_queries)


def epoch_acc(model, batch_size, sql_data, table_data, save_results=False):
    '''

    '''
    model.eval()
    perm = list(range(len(sql_data)))
    start = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while start < len(sql_data):
        end = start + batch_size if start + \
            batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, ground_truth_cond_seq, raw_data =\
            gen_batch_sequence(sql_data, table_data, perm, start, end)

        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]

        query_gt, table_ids = generate_batch_query(sql_data, perm, start, end)
        ground_truth_sel_seq = [x[1] for x in ans_seq]

        score = model.forward(q_seq, col_seq, )
        pred_queries = model.gen_query(score, q_seq, col_seq,
                                       raw_q_seq, raw_col_seq)
        one_err, tot_err = model.check_accuracy(pred_queries, query_gt)

        one_acc_num += (end - start - one_err)
        tot_acc_num += (end - start - tot_err)

        start = end

    return tot_acc_num / len(sql_data), one_acc_num/len(sql_data)
