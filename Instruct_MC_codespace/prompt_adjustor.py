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

# CSQA data
CSQA_data = []
with open (CSQA_PATH, "r", encoding="utf-8") as f:
    CSQA_data = json.load(f)
CSQA_length = len(CSQA_data)

data = CSQA_data
data_length = CSQA_length
split_ratio = None

try:
    train_all_result, train_gennal_marker_path = tt.train("CSQA", data, split_ratio)
    score_list = tt.test("CSQA", data, split_ratio, train_gennal_marker_path)
    print("CSQA all result:", score_list)

except Exception as e:
    print("Error: ", str(e))
    traceback.print_exc()
    pass