import json
import logging
import os
import shutil
from sys import maxsize
from time import localtime, strftime, time


def makedirs(directory):
    if isinstance(directory, str):
        directory = [directory]
    
    for d in directory:
        if '.' in d:
            d = '/'.join(d.split('/')[:-1])
        if not os.path.isdir(d):
            os.makedirs(d)

def copy(source: str, target: str):
    if os.path.isfile(source):
        if os.path.isdir(target):
            shutil.copy(source, target + f'/{source.split("/")[-1]}')
        else:
            shutil.copy(source, target)
    elif os.path.isdir(source):
        shutil.copytree(source, target)

def init_logging(log_path):
    log_dir = '/'.join(log_path.split('/')[:-1])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    log_format = '[ %(asctime)s ] %(message)s'
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode='w', encoding='UTF-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(sh)

def load_jsonl(file_path):
    # load file
    file_name = file_path.split('/')[-1]
    if file_name.endswith('.json'):
        file_name = file_name.replace('.json', '', 1)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.endswith('.jsonl'):
        file_name = file_name.split('/')[-1].replace('.jsonl', '', 1)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
        
    return data
    
def save_jsonl(data, save_path, mode='w'): # save_path = f'./res/{strftime('%Y-%m-%d %H-%M-%S')}.jsonl'
    with open(save_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        f.flush()
