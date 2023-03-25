import json
from types import GeneratorType
from datetime import datetime

import numpy as np
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
            sql = json.loads(line.strip()) #remove white space before and after 
            sql_data.append(sql)
            sql_query_count += 1
    with open(tables_file) as lines:
        for line in lines:
            tab = json.loads(line.strip())
            table_data[tab['id']] = tab
            table_count += 1
    print(f"Loaded {sql_query_count} queries and {table_count} tables")
    return sql_data,table_data



def gen_batch_sequence(sql_data, table_data,idxes,start, end):
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
        raw_data.append((sql['question'],table_data[table_id]['header'], sql['query']))
        
    return (question_seq, column_seq, number_of_col, answer_seq, query_seq, ground_truth_cond_seq,raw_data)


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
    ret_h, ret_c = lstm(inp, prev_hidden)
    return ret_h, ret_c


def generate_batch_query(sql_data, idx, start, end):
# TODO: this fucntion is redundant please find a proper alternative by gen_batch sequence.
    '''
    Input: 
        sql_data -  a list of sql query dictionary [this is the original data and not shuffled]
        idx - the order in which the dataset is shuffled
        start - start index
        end - end index
    
    Output: 
        query_gt: lsit containing the ground truth sql query of the batch
        table_id: list containing the tables sql queries.
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
        toy_tables[toy_queries[x]['table_id']] = table_data[toy_queries[x]['table_id']]
    return toy_queries, toy_tables
