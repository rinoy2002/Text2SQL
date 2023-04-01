<<<<<<< HEAD
import threading
import sys, os
import logging
import pandas as pd

def annotate_data_set(dataset_name):
    queries_file_name = './data/'+dataset_name+'.jsonl'
    
    if not os.path.exists(queries_file_name):
        logging.error(f"file not found {dataset_name}")
    
    queries_dataframe = pd.read_json(queries_file_name,lines=True)
=======
import threading
import sys, os
import logging
import pandas as pd

def annotate_data_set(dataset_name):
    queries_file_name = './data/'+dataset_name+'.jsonl'
    
    if not os.path.exists(queries_file_name):
        logging.error(f"file not found {dataset_name}")
    
    queries_dataframe = pd.read_json(queries_file_name,lines=True)
>>>>>>> refs/rewritten/Add-gen-col-batch
    print(queries_dataframe.head())