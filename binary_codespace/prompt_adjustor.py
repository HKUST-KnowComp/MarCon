import json
from openai import AzureOpenAI
from constants import *
import Llama_response as llm
import GenNal
import train_test as tt
import time
import os
import sys
import tqdm
import re
# import marker2graph as m2g
import traceback

if(os.path.exists(OUTPUT_DATA_DIR) == False):
    os.makedirs(OUTPUT_DATA_DIR)
print("output dir:", OUTPUT_DATA_DIR)

# BoolQ data
BoolQ_data = []
BoolQ_train = []
with open (BOOLQ_TRAIN_PATH, "r", encoding = "utf-8") as f:
    print("Reading BoolQ training data...")
    for line in f:
        BoolQ_data.append(json.loads(line))
        BoolQ_train.append(json.loads(line))
BoolQ_train_length = len(BoolQ_train)
BoolQ_dev = []
with open (BOOLQ_DEV_PATH, "r", encoding="utf-8") as f:
    print("Reading BoolQ dev data...")
    for line in f:
        BoolQ_data.append(json.loads(line))
        BoolQ_dev.append(json.loads(line))
BoolQ_dev_length = len(BoolQ_dev)
BoolQ_length = len(BoolQ_data)

# import your custom dataset
"""
with open("your_dataset.json", "r", encoding="utf-8") as f:
    print("Reading your dataset...")
    for line in f:
        your_data.append(json.loads(line))
"""

# specify the split ratio
split_ratio = BoolQ_train_length / (BoolQ_train_length + BoolQ_dev_length)

try:
    BoolQ_all_result, BoolQ_gennal_marker_path = tt.train("BoolQ", BoolQ_data, split_ratio)
    score_list = tt.test("BoolQ", BoolQ_data, split_ratio, BoolQ_gennal_marker_path)
    print("BoolQ all result:", score_list)

    
except Exception as e:
    print("Error: ", str(e))
    traceback.print_exc()
    pass