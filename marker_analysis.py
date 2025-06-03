import json
import os
import numpy as np
import re
from scipy.stats import spearmanr
import itertools
import marker_analysis_tool as func

model_avg_spearman = {} 

all_models = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct", 
    "Qwen2.5-14B-Instruct", 
    "Qwen2.5-32B-Instruct", 
    "Mistral-7B-Instruct-v0.3", 
    "gpt-4o", 
    "gpt-4o-mini"
]
all_datasets = [
    "BoolQ", 
    "StrategyQA", 
    "GSM8K", 
    "MMLU", 
    "CSQA", 
    "MedMCQA", 
    "CaseHOLD",
]
marker_num = 10
marker_count = 10
filter_threshold = marker_num
ratio = 0.05
mode = "marker_count"
base_path = "your base path here"  # Replace with your actual base path
output_path = "{}/all_markers_thres={}.json".format(base_path, marker_count)

all_marker_dic = {}
all_number_dic = {}
all_marker_acc_dic = {}
for model_name in all_models:
    print("Processing:", model_name)
    model_marker_dic = {}
    model_number_dic = {}
    model_marker_acc_dic = {}
    for dataset_name in all_datasets:
        print("\tProcessing:", dataset_name)
        if(dataset_name == "GSM8K" and model_name == "Qwen2.5-32B-Instruct"):
            train_end_index = 7468
            marker_path = base_path + f"{dataset_name}_results/F_{model_name}/{dataset_name}_gennal_filtered_markers_single_0~{train_end_index}.json"
            train_path = base_path + f"{dataset_name}_results/F_{model_name}/{dataset_name}_gennal_0~{train_end_index}.json"
            with open(marker_path, 'r', encoding='utf-8') as f:
                marker_data = json.load(f)[0]
            with open(train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            all_correctness = []
            original_marker_data = marker_data.copy()
            for qa_dic in train_data:
                correctness = qa_dic["single_round_gennal"][0]["correctness"]
                if(correctness != None):
                    all_correctness.append(correctness)
            marker_acc = np.mean(np.array(all_correctness))
            model_marker_acc_dic[dataset_name] = marker_acc
        else:
            if(dataset_name == "BoolQ"):
                train_end_indices = [9427]
            elif(dataset_name == "StrategyQA"):
                train_end_indices = [2061]
            elif(dataset_name == "GSM8K"):
                train_end_indices = [7468]
            elif(dataset_name == "MMLU"):
                train_end_indices = [20000]
            elif(dataset_name == "CSQA"):
                train_end_indices = [8769]
            elif(dataset_name == "MedMCQA"):
                train_end_indices = [9686]
            elif(dataset_name == "CaseHOLD"):
                train_end_indices = [8396]
            train_end_index = -1
            for train_index in train_end_indices:
                marker_path = base_path + f"{dataset_name}_results/F_{model_name}/{dataset_name}_gennal_filtered_markers_single_0~{train_index}.json"
                number_path = base_path + f"{dataset_name}_results/F_{model_name}/{dataset_name}_gennal+clanal+gennum_0~{train_index}.json"
                if(not os.path.exists(marker_path) or not os.path.exists(number_path)):
                    continue
                train_end_index = train_index
                break
            if(train_end_index == -1):
                print("\t\tNo marker file found for {}~{}".format(dataset_name, model_name))
                continue
            
            with open(marker_path, 'r', encoding='utf-8') as f:
                marker_data = json.load(f)[0]
            with open(number_path, 'r', encoding='utf-8') as f:
                number_data = json.load(f)
        
        all_markers = []
        
        for marker, marker_list in marker_data.items():
            for marker_data in marker_list:
                all_markers.append((marker, marker_data["marker_count"], marker_data["marker_correct_ratio"]))

    all_marker_dic[model_name] = model_marker_dic
    # all_number_dic[model_name] = model_number_dic
    all_marker_acc_dic[model_name] = model_marker_acc_dic


with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_marker_dic, f, indent=4, ensure_ascii=False)
with open(output_path.replace("marker", "marker_acc"), 'w', encoding='utf-8') as f:
    json.dump(all_marker_acc_dic, f, indent=4, ensure_ascii=False)
    
with open(output_path, 'r', encoding='utf-8') as f:
    all_marker_dic = json.load(f)
with open(output_path.replace("marker", "marker_acc"), 'r', encoding='utf-8') as f:
    all_marker_acc_dic = json.load(f)
with open("{}/all_cvs_thres={}.json".format(filter_threshold), 'r', encoding='utf-8') as f:
    all_cvs = json.load(f)
with open("{}/all_markers_thres={}.json".format(filter_threshold), 'r', encoding='utf-8') as f:
    all_marker_acc_dic = json.load(f)
        
func.spearman_correlation(all_marker_dic, marker_count)
func.calculate_concentration(all_marker_dic, marker_count)
func.calculate_concentration_number(all_number_dic, marker_count)
func.calculate_dataset_avg_cv(all_marker_dic, marker_count)
func.calculate_acc_cv_cv(all_cvs, all_marker_acc_dic, marker_count)
func.calculate_model_marker_avgcv(all_marker_dic, marker_count)

corr_abs, corr_rank = func.compute_model_stability_correlations(all_marker_acc_dic, all_marker_dic)
print("Correlation between models:", corr_abs, corr_rank)

func.calculate_marker_model_correlation(all_marker_dic, all_marker_acc_dic)