import Llama_response as llm
import os
import json
# import matplotlib.pyplot as plt
from constants import *
import numpy as np
import pandas as pd
import re
import tqdm
import marker2graph as m2g
import GenNum
import ClaNal
import GenNal
from netcal.metrics import ECE, MCE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve



def ace(y_true, y_prob, n_bins=ACE_BINS):
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)

    ace = np.sum(np.abs(fraction_of_positives - mean_predicted_value)) / n_bins
    return ace

def train(dataset_name, data, split_ratio, start_index=None, end_index=None, check_exist=True):
    filter_threshold = TRAIN_FILTER_THRESHOLD
    if(start_index == None and end_index == None):
        length = len(data)
        start_index = 0
        end_index = int(length * split_ratio)
        if(check_exist == True):
            train_output_single_marker_path = OUTPUT_DATA_DIR + f"{dataset_name}_markers_{start_index}~{end_index}.json"
            output_gennal_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_{start_index}~{end_index}.json"
            new_marker_range = f"{start_index}~{end_index}"
            if(os.path.exists(train_output_single_marker_path) == False):
                single_filtered_marker_path, double_filtered_marker_path, gennal_result_path = GenNal.All_gennal(data, start_index, end_index, dataset_name, filter_threshold)
            else:
                single_filtered_marker_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_filtered_markers_single_{new_marker_range}.json"
                double_filtered_marker_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_filtered_markers_double_{new_marker_range}.json"
                gennal_result_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_{new_marker_range}.json"
        else:
            single_filtered_marker_path, double_filtered_marker_path, gennal_result_path = GenNal.All_gennal(data, start_index, end_index, dataset_name, filter_threshold)
    else:
        length = end_index - start_index
        end_index = int(start_index + split_ratio * length)
        single_filtered_marker_path, double_filtered_marker_path, gennal_result_path = GenNal.All_gennal(data, start_index, end_index, dataset_name, filter_threshold)    
    with open(gennal_result_path, 'r', encoding='utf-8') as f:
        gennal_results = json.load(f)
    return gennal_results, single_filtered_marker_path




def test(dataset_name, data, split_ratio, trained_gennal_marker_mapping_path, start_index=None, end_index=None):
    
    
    filter_threshold = TEST_FILTER_THRESHOLD
    # first determine the start and end index
    if(start_index == None and end_index == None):
        length = len(data)
        start_index = int(length * split_ratio)
        end_index = length - 1
    else:
        length = end_index - start_index
        start_index = int(start_index + split_ratio * length)
    
    
    # get the marker mapping
    with open(trained_gennal_marker_mapping_path, 'r', encoding='utf-8') as f:
        gennal_marker_mapping = json.load(f)
    
    print("Generating test-time answers ...")
    test_gennal_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_{start_index}~{end_index}.json"
    if(os.path.exists(test_gennal_path) == False):
        gennal_single_marker_path, gennal_double_marker_path, gennal_result_path = GenNal.All_gennal(data, start_index, end_index, dataset_name, filter_threshold)
    else:
        gennal_single_marker_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_filtered_markers_single_{start_index}~{end_index}.json"
        gennal_result_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_{start_index}~{end_index}.json"
        
    # @ all_result_path is the test-time all results
    with open(gennal_result_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    gen_gen_prediction = []
    gen_gen_gold_answers = []
    
    all_numbers = len(all_results)
    valid_numbers = len(all_results)
    for answer_dic in all_results:
        marker = answer_dic["single_round_gennal"][0]["epistemic_markers"]
        gennal_marker_dic = gennal_marker_mapping[0]
        # @ gen_found: this marker is in the gennal mapping
        gen_found = False
        for mapped_marker in gennal_marker_dic:
            if(marker == mapped_marker):
                gen_found = True
                break
        if(gen_found == False): 
            print(f"Error: marker not found in the mapping, index {answer_dic['index']}")
            valid_numbers -= 1
            continue
        else: # gen_found == True 
            if((gennal_marker_dic[mapped_marker][0]["marker_correct_ratio"] == None) or 
               (answer_dic['single_round_gennal'][0]['correctness'] == None)):
                valid_numbers -= 1
                continue
            gen_gen_prediction.append(gennal_marker_dic[mapped_marker][0]["marker_correct_ratio"])
            gen_gen_gold_answers.append(answer_dic['single_round_gennal'][0]['correctness'])

    # then use package in netcal to calculate the calibration error
    gen_gen_gold_answers = np.array(gen_gen_gold_answers)
    gen_gen_prediction = np.array(gen_gen_prediction)
    ECE_BINS = len(gen_gen_gold_answers)
    ACE_BINS = len(gen_gen_gold_answers)
    MCE_BINS = len(gen_gen_gold_answers)
    
    
    gen_gen_ece = ECE(bins=ECE_BINS)
    gen_gen_mce = MCE(bins=MCE_BINS)
    gen_gen_brier_score = brier_score_loss(gen_gen_gold_answers, gen_gen_prediction)
    gen_gen_auroc_score = roc_auc_score(gen_gen_gold_answers, gen_gen_prediction)
    gen_gen_precision, gen_gen_recall, _ = precision_recall_curve(gen_gen_gold_answers, gen_gen_prediction)
    gen_gen_auprc_score = auc(gen_gen_recall, gen_gen_precision)
    gen_gen_ece_score = gen_gen_ece.measure(gen_gen_prediction, gen_gen_gold_answers)
    gen_gen_mce_score = gen_gen_mce.measure(gen_gen_prediction, gen_gen_gold_answers)
    gen_gen_ace_score = ace(gen_gen_gold_answers, gen_gen_prediction, n_bins=ACE_BINS)
    gen_gen_acc = np.sum(gen_gen_gold_answers) / len(gen_gen_gold_answers)    
    
    score_dic = {
                "gen_gen": {
                        "ece": gen_gen_ece_score, "mce": gen_gen_mce_score, "accuracy": gen_gen_acc, "brier_score": gen_gen_brier_score, "auroc": gen_gen_auroc_score, "auprc": gen_gen_auprc_score, "ace": gen_gen_ace_score
                    }
            }

    method_calibration_error_path = OUTPUT_DATA_DIR + dataset_name + "_calibration_error.json"
    with open(method_calibration_error_path, 'w', encoding='utf-8') as f:
        json.dump(score_dic, f, ensure_ascii=False, indent=4)

    return score_dic