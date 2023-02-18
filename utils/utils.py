import json

def load_data_set(dataset_name):
    """
    Universal function to load the dataset
    Input: dataset_name - str {train, test, dev}
    Output: (sql_data, table_data) - (list, dictionary): containg the tokenized queries and the table dataset.
    """
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
    This function takes in 3 arguments 
    lstm : the name of the lstm variable that needs to be run
    inp  : the input in the for [Batch size , num_tok, last_layer]
    inp_length: an array that contains the length of each element in the batch size = batch size
    pre_hidden: hidden layer values of the previous lstm layer
    '''
    ret_h, ret_c = lstm(inp, prev_hidden)
    return ret_h, ret_c
