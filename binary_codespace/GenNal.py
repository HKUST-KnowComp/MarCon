from openai import AzureOpenAI
from constants import *
import Llama_response as llm
import tqdm
import traceback
import re
# import marker2graph as m2g
Gennal_temperature = 0

def remove_non_alphanumeric(s):
    return re.sub(r'[^a-zA-Z0-9\s]', '', s)

def find_first_yes_no(text):
    match = re.search(r'##(yes|no)##', text, re.IGNORECASE)
    if match:
        return remove_non_alphanumeric(match.group(0))
    else:
        return None

# this function collects the gennal responses
def GenNal(QA_pairs, start_index, end_index, dataset_name):
    # specify the output path
    output_gennal_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_{start_index}~{end_index}.json"
    # initialize the list to store the answers
    answers = []
    # initialize the list to store the error indexes
    error_list = []
    # initialize the list to store the epistemic markers
    single_round_marker_list = {}
    double_round_marker_list = {}
    print("Generating Gennal responses...")
    for i in tqdm.tqdm(range(start_index, end_index)):
        
        
        try:
            # initialize relevant information
            i_dic = {}
            i_dic['index'] = i
            i_dic['question'] = QA_pairs[i]['question']
            i_dic['gold_answer'] = QA_pairs[i]['answer']
            i_dic['single_round_gennal'] = []
            i_dic['double_round_gennal'] = []
            i_dic["error"] = "no error"
            
            
            # start implementing gennal process
            # get response
            # single round gennal
            question = QA_pairs[i]['question']
            answer = QA_pairs[i]['answer']
            single_round_gennal = llm.get_response(GENNAL_PROMPT_SINGLE + question + GENNAL_PROMPT_INDUCE, INSTRUCTION)
            i_dic["single_round_gennal"].append({"full_response": single_round_gennal})
            # double round gennal (not implemented in the final version of paper)
            # binary_answer = llm.get_response(GENNAL_PROMPT_DOUBLE_1 + question, INSTRUCTION)
            binary_answer = None
            i_dic['double_round_gennal'].append({"binary_answer": binary_answer})
            # double_round_gennal = llm.get_response(GENNAL_PROMPT_DOUBLE_2 + question + GENNAL_PROMPT_DOUBLE_3 + binary_answer, INSTRUCTION)
            double_round_gennal = None
            i_dic['double_round_gennal'][0]["full_response"] = double_round_gennal
            
            
            # extract the epistemic markers
            # do it on the single_round_gennal first
            if(i_dic['single_round_gennal'][0]['full_response'] == None):
                error_list.append({"error_index": i, "error_type": "single", "error_message": "None response"})
                i_dic['single_round_gennal'][0]["epistemic_markers"] = None
                i_dic['error'] = "no response"
            else:
                # set the epistemic markers for the single_round_gennal
                # chat template version
                SINGLE_TEMPLATE.append({"role": "user", "content": i_dic['single_round_gennal'][0]['full_response']})
                epistemic_markers = llm.get_response(None, INSTRUCTION, SINGLE_TEMPLATE)
                SINGLE_TEMPLATE.pop()
                
                
                match = re.search(r"\*\*(.*?)\*\*", epistemic_markers)
                if match:
                    epistemic_markers = match.group(1) 
                    i_dic['single_round_gennal'][0]["epistemic_markers"] = epistemic_markers.lower()
                else:
                    single_round_gennal_backup_1 = llm.get_response(GENNAL_PROMPT_SINGLE + question, INSTRUCTION)
                    i_dic['single_round_gennal'][0]["full_response"] = single_round_gennal_backup_1
                
                    # chat template version
                    SINGLE_TEMPLATE.append({"role": "user", "content": i_dic['single_round_gennal'][0]['full_response']})
                    epistemic_markers = llm.get_response(None, INSTRUCTION, SINGLE_TEMPLATE)
                    SINGLE_TEMPLATE.pop()
                    
                    
                    match = re.search(r"\*\*(.*?)\*\*", epistemic_markers)
                    if match:
                        epistemic_markers = match.group(1) 
                        i_dic['single_round_gennal'][0]["epistemic_markers"] = epistemic_markers.lower()
                    else:
                        single_round_gennal_backup_2 = llm.get_response(GENNAL_PROMPT_SINGLE + question, INSTRUCTION)
                        i_dic['single_round_gennal'][0]["full_response"] = single_round_gennal_backup_2
                        
                        
                        SINGLE_TEMPLATE.append({"role": "user", "content": i_dic['single_round_gennal'][0]['full_response']})
                        epistemic_markers = llm.get_response(None, INSTRUCTION, SINGLE_TEMPLATE)
                        SINGLE_TEMPLATE.pop()
                        
                        
                        match = re.search(r"\*\*(.*?)\*\*", epistemic_markers)
                        if match:
                            epistemic_markers = match.group(1) 
                            i_dic['single_round_gennal'][0]["epistemic_markers"] = epistemic_markers.lower()
                        else:
                            single_round_gennal_backup_3 = llm.get_response(GENNAL_PROMPT_SINGLE + question, INSTRUCTION)
                            i_dic['single_round_gennal'][0]["full_response"] = single_round_gennal_backup_3
                            
                            
                            SINGLE_TEMPLATE.append({"role": "user", "content": i_dic['single_round_gennal'][0]['full_response']})
                            epistemic_markers = llm.get_response(None, INSTRUCTION, SINGLE_TEMPLATE)
                            SINGLE_TEMPLATE.pop()
                            
                            
                            match = re.search(r"\*\*(.*?)\*\*", epistemic_markers)
                            if match:
                                epistemic_markers = match.group(1) 
                                i_dic['single_round_gennal'][0]["epistemic_markers"] = epistemic_markers.lower()
                            else:
                                single_round_gennal_backup_4 = llm.get_response(GENNAL_PROMPT_SINGLE + question, INSTRUCTION)
                                i_dic['single_round_gennal'][0]["full_response"] = single_round_gennal_backup_4
                                
                                
                                SINGLE_TEMPLATE.append({"role": "user", "content": i_dic['single_round_gennal'][0]['full_response']})
                                epistemic_markers = llm.get_response(None, INSTRUCTION, SINGLE_TEMPLATE)
                                SINGLE_TEMPLATE.pop()
                                
                                match = re.search(r"\*\*(.*?)\*\*", epistemic_markers)
                                if match:
                                    epistemic_markers = match.group(1) 
                                    i_dic['single_round_gennal'][0]["epistemic_markers"] = epistemic_markers.lower()
                                else:
                                    single_round_gennal_backup_5 = llm.get_response(GENNAL_PROMPT_SINGLE + question, INSTRUCTION)
                                    i_dic['single_round_gennal'][0]["full_response"] = single_round_gennal_backup_5
                                    
                                    
                                    SINGLE_TEMPLATE.append({"role": "user", "content": i_dic['single_round_gennal'][0]['full_response']})
                                    epistemic_markers = llm.get_response(None, INSTRUCTION, SINGLE_TEMPLATE)
                                    SINGLE_TEMPLATE.pop()
                                    
                                    match = re.search(r"\*\*(.*?)\*\*", epistemic_markers)
                                    if match:
                                        epistemic_markers = match.group(1) 
                                        i_dic['single_round_gennal'][0]["epistemic_markers"] = epistemic_markers.lower()
                                    else:
                                        i_dic['single_round_gennal'][0]["epistemic_markers"] = None
                                        i_dic["error"] = "no marker"
            if(i_dic['single_round_gennal'][0]["epistemic_markers"] != None):
                i_dic['single_round_gennal'][0]["epistemic_markers"] = remove_non_alphanumeric(i_dic['single_round_gennal'][0]["epistemic_markers"])
                                        
            # then do it on the double_round_gennal
            # do exception handling first
            if(i_dic['double_round_gennal'][0]['binary_answer'] == None):
                error_list.append({"error_index": i, "error_type": "double", "error_message": "None binary answer"})
                i_dic['double_round_gennal'][0]["epistemic_markers"] = None
            elif(i_dic['double_round_gennal'][0]['full_response'] == None):
                error_list.append({"error_index": i, "error_type": "double", "error_message": "None full text response"})
                i_dic['double_round_gennal'][0]["epistemic_markers"] = None
            else:
                epistemic_markers = None
                match = re.search(r"\*\*(.*?)\*\*", epistemic_markers)
                if match:
                    epistemic_markers = match.group(1) 
                    i_dic['double_round_gennal'][0]["epistemic_markers"] = epistemic_markers.lower()
                else:
                    epistemic_markers = None
                    i_dic['double_round_gennal'][0]["epistemic_markers"] = None
    
                
            # judge whether each question is right
            # single round gennal first
            single_binary_answer = find_first_yes_no(i_dic['single_round_gennal'][0]['full_response'])
            print("single round gennal answer:", single_binary_answer)
            single_binary_answer = single_binary_answer.lower() if single_binary_answer != None else None
            i_dic['single_round_gennal'][0]['binary_answer'] = single_binary_answer
            double_binary_answer = None
            double_binary_answer = double_binary_answer.lower() if double_binary_answer != None else None
            if(i_dic['single_round_gennal'][0]['full_response'] == None
               or single_binary_answer == None):
                print("correctness is none since no response or no binary answer detected.")
                i_dic['single_round_gennal'][0]["correctness"] = None
                i_dic['error'] = "no response or no binary answer detected"
            elif((single_binary_answer == "yes" and i_dic['gold_answer'] == True) or 
                (single_binary_answer == "no" and i_dic['gold_answer'] == False)):
                i_dic['single_round_gennal'][0]["correctness"] = 1
            elif((single_binary_answer == "yes" and i_dic['gold_answer'] == False) or
                (single_binary_answer == "no" and i_dic['gold_answer'] == True)):
                i_dic['single_round_gennal'][0]["correctness"] = 0
            else:
                print("correctness is none since binary answer detected is neither yes nor no.")
                i_dic['single_round_gennal'][0]["correctness"] = None
            # then double round gennal (could be ignored)
            if(i_dic['double_round_gennal'][0]['binary_answer'] == None):
                i_dic['double_round_gennal'][0]["correctness"] = None
            elif((double_binary_answer == "yes" and i_dic['gold_answer'] == True) or
                (double_binary_answer == "no" and i_dic['gold_answer'] == False)):
                i_dic['double_round_gennal'][0]["correctness"] = 1
            elif((double_binary_answer == "no" and i_dic['gold_answer'] == True) or
                (double_binary_answer == "yes" and i_dic['gold_answer'] == False)):
                i_dic['double_round_gennal'][0]["correctness"] = 0
            else:
                i_dic['double_round_gennal'][0]["correctness"] = None
                
                
            # add the markers into the marker list
            # single round gennal first        
            if(i_dic['single_round_gennal'][0]["epistemic_markers"] != None):
                if(i_dic['single_round_gennal'][0]['correctness'] !=  None):
                    if(i_dic['single_round_gennal'][0]["epistemic_markers"] not in single_round_marker_list):
                        # if the marker is not in the list, add it to the list
                        single_round_marker_list[i_dic['single_round_gennal'][0]["epistemic_markers"]] = []
                        single_round_marker_list[i_dic['single_round_gennal'][0]["epistemic_markers"]].append({"marker_count": 1, "marker_correct_count": 0})
                    else:
                        # if the marker is in the list, add the count by 1
                        single_round_marker_list[i_dic['single_round_gennal'][0]["epistemic_markers"]][0]['marker_count'] += 1 
                    single_round_marker_list[i_dic['single_round_gennal'][0]["epistemic_markers"]][0]['marker_correct_count'] += i_dic['single_round_gennal'][0]['correctness']
                else:
                    # If the binary answer is not properly detected, pass
                    i_dic['error'] = "binary answer detection error"
                    pass
            else:
                # if the marker is None, pass
                i_dic['error'] = "no marker"
                pass
            # then double round gennal
            if(i_dic['double_round_gennal'][0]["epistemic_markers"] != None):
                if(i_dic['double_round_gennal'][0]['correctness'] != None):
                    if(i_dic['double_round_gennal'][0]['epistemic_markers'] not in double_round_marker_list):
                        # if the marker is not in the list, add it to the list
                        double_round_marker_list[i_dic['double_round_gennal'][0]['epistemic_markers']] = []
                        double_round_marker_list[i_dic['double_round_gennal'][0]['epistemic_markers']].append({"marker_count": 1, "marker_correct_count": 0})
                    else:
                        # if the marker is in the list, add the count by 1
                        double_round_marker_list[i_dic['double_round_gennal'][0]['epistemic_markers']][0]['marker_count'] += 1
                    # then judge if the response with the marker is correct
                    double_round_marker_list[i_dic['double_round_gennal'][0]['epistemic_markers']][0]['marker_correct_count'] += i_dic['double_round_gennal'][0]['correctness']
                else:
                    # If the binary answer is not properly detected, pass
                    i_dic['error'] = "binary answer detection error"
                    pass
            else:
                # if the marker is None, pass
                pass  
            # add the data into the answers list
            answers.append(i_dic)
            
            
        except Exception as e:
            print("exception caught!")
            error_message = str(e)
            traceback.print_exc()
            print(error_message)
            if "TypeError" in error_message:
                i_dic["error"] = "GenNal type error"
                answers.append(i_dic)
                continue
            else:
                i_dic["error"] = "GenNal other error"
                error_message = str(e)
                print(error_message)
                answers.append(i_dic)
                # continue
                raise
    # with open(OUTPUT_DATA_DIR + "retry_list.json", "w", encoding='utf-8') as f:
       # json.dump(retry_list, f, ensure_ascii=False, indent=4)
        
        
    with open(output_gennal_path, "w", encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    return answers, output_gennal_path
    
    
        

# this function turns the gennal response into marker statistics
def marker_counter(dataset_name,datapath=None, data=None):
    
    
    # determine what data is
    if(data != None):
        pass
    else:
        if(datapath != None):
            with open(datapath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print("please check your code. No readable data.")
            raise
        
    
    # specify the marker range
    start_index = data[0]['index']
    end_index = data[-1]['index'] + 1
    
    
    # specify the marker path
    output_single_marker_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_markers_single_{start_index}~{end_index}.json"
    output_double_marker_path = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_markers_double_{start_index}~{end_index}.json"
    
    
    # data = data[0] # this is strange, i don't know why the data is a list of a list
    print("Calculating the markers statistics...")
    single_round_marker_list = {}
    double_round_marker_list = {}
    print("preparing the marker list...")
    for i in tqdm.tqdm(range(len(data))):
        # add the markers into the marker list
        # single round gennal first 
        
        if(data[i]['single_round_gennal'][0]["epistemic_markers"] != None
           and data[i]["error"] == "no error"):
            if(data[i]['single_round_gennal'][0]["epistemic_markers"] not in single_round_marker_list):
                # if the marker is not in the list, add it to the list
                single_round_marker_list[data[i]['single_round_gennal'][0]["epistemic_markers"]] = []
                single_round_marker_list[data[i]['single_round_gennal'][0]["epistemic_markers"]].append({"marker_count": 1, "marker_correct_count": 0})
            else:
                # if the marker is in the list, add the count by 1
                single_round_marker_list[data[i]['single_round_gennal'][0]["epistemic_markers"]][0]['marker_count'] += 1 
            single_round_marker_list[data[i]['single_round_gennal'][0]["epistemic_markers"]][0]['marker_correct_count'] += data[i]['single_round_gennal'][0]['correctness']
        else:
            # if the marker is None, pass
            pass
        # then double round gennal
        if(data[i]['double_round_gennal'][0]["epistemic_markers"] != None
           and data[i]["error"] == "no error"):
            if(data[i]['double_round_gennal'][0]['epistemic_markers'] not in double_round_marker_list):
                # if the marker is not in the list, add it to the list
                double_round_marker_list[data[i]['double_round_gennal'][0]['epistemic_markers']] = []
                double_round_marker_list[data[i]['double_round_gennal'][0]['epistemic_markers']].append({"marker_count": 1, "marker_correct_count": 0})
            else:
                # if the marker is in the list, add the count by 1
                double_round_marker_list[data[i]['double_round_gennal'][0]['epistemic_markers']][0]['marker_count'] += 1
            # then judge if the response with the marker is correct
            double_round_marker_list[data[i]['double_round_gennal'][0]['epistemic_markers']][0]['marker_correct_count'] += data[i]['double_round_gennal'][0]['correctness']
        else:
            # if the marker is None, pass
            pass  
        
       
    # save the marker statistics in the data 
    print("saving marker statistics...")
    for i in tqdm.tqdm(range(len(data))): 
        # single round gennal
        if(data[i]['single_round_gennal'][0]['epistemic_markers'] in single_round_marker_list):
            data[i]['single_round_gennal'][0]['marker_count'] = single_round_marker_list[data[i]['single_round_gennal'][0]['epistemic_markers']][0]['marker_count']
            data[i]['single_round_gennal'][0]["marker_correct_count"] = single_round_marker_list[data[i]['single_round_gennal'][0]['epistemic_markers']][0]['marker_correct_count']
            data[i]['single_round_gennal'][0]["marker_correct_ratio"] = data[i]['single_round_gennal'][0]["marker_correct_count"] / data[i]['single_round_gennal'][0]["marker_count"]
        else:
            print(f"please check your code. index {i} raised a problem.")
        # double round gennal
        if(data[i]['double_round_gennal'][0]['epistemic_markers'] in double_round_marker_list):
            data[i]['double_round_gennal'][0]['marker_count'] = double_round_marker_list[data[i]['double_round_gennal'][0]['epistemic_markers']][0]['marker_count']
            data[i]['double_round_gennal'][0]["marker_correct_count"] = double_round_marker_list[data[i]['double_round_gennal'][0]['epistemic_markers']][0]['marker_correct_count']
            data[i]['double_round_gennal'][0]["marker_correct_ratio"] = data[i]['double_round_gennal'][0]["marker_correct_count"] / data[i]['double_round_gennal'][0]["marker_count"]
        else:
            # print(f"please check your code. index {i} raised a problem.")
            pass
            
            
    # save the correct ratio in the marker list
    for marker in single_round_marker_list:
        single_round_marker_list[marker][0]["marker_correct_ratio"] = single_round_marker_list[marker][0]["marker_correct_count"] / single_round_marker_list[marker][0]["marker_count"]
    for marker in double_round_marker_list:
        double_round_marker_list[marker][0]["marker_correct_ratio"] = double_round_marker_list[marker][0]['marker_correct_count'] / double_round_marker_list[marker][0]["marker_count"]
        
    
    # sort the correct ratio
    temp_single_marker_dic = {}
    temp_double_marker_dic = {}
    for marker, value_list in single_round_marker_list.items():
        temp_single_marker_dic[marker] = value_list[0]["marker_correct_ratio"]
    for marker, value_list in double_round_marker_list.items():
        temp_double_marker_dic[marker] = value_list[0]["marker_correct_ratio"] 
    temp_single_marker_dic = dict(sorted(temp_single_marker_dic.items(), key=lambda x: x[1]))
    temp_double_marker_dic = dict(sorted(temp_double_marker_dic.items(), key=lambda x: x[1]))
    final_single_marker_dic = {}
    final_double_marker_dic = {}
    for ordered_marker in temp_single_marker_dic:
        for random_marker in single_round_marker_list:
            if(random_marker == ordered_marker):
                final_single_marker_dic[ordered_marker] = single_round_marker_list[random_marker]
    for ordered_marker in temp_double_marker_dic:
        for random_marker in double_round_marker_list:
            if(random_marker == ordered_marker):
                final_double_marker_dic[ordered_marker] = double_round_marker_list[random_marker]
    # single_round_marker_list = dict(sorted(single_round_marker_list.items(), key = lambda x: x[1][0]['marker_correct_count'], reverse=True))
    # double_round_marker_list = dict(sorted(double_round_marker_list.items(), key = lambda x: x[1][0]['marker_correct_count'], reverse=True))
    
    
    # save the index that the markers appear for gennum use
    print("Saving the marker index...")
    for answer_dic in tqdm.tqdm(data):
        for marker in single_round_marker_list:
            if(answer_dic['single_round_gennal'][0]['epistemic_markers'] == marker):
                if('marker_index' in single_round_marker_list[marker][0]):
                    single_round_marker_list[marker][0]['marker_index'].append(answer_dic['index'])
                else:
                    single_round_marker_list[marker][0]['marker_index'] = [answer_dic['index']]
        for marker in double_round_marker_list:
            if(answer_dic['double_round_gennal'][0]['epistemic_markers'] == marker):
                if('marker_index' in double_round_marker_list[marker][0]):
                    double_round_marker_list[marker][0]['marker_index'].append(answer_dic['index'])
                else:
                    double_round_marker_list[marker][0]['marker_index'] = [answer_dic['index']]
                    
                    
    # save the data
    with open (output_single_marker_path, "w", encoding='utf-8') as f:
        json.dump([single_round_marker_list], f, ensure_ascii=False, indent=4)
    with open (output_double_marker_path, "w", encoding='utf-8') as f:
        json.dump([double_round_marker_list], f, ensure_ascii=False, indent=4)
    return output_single_marker_path, output_double_marker_path
        
        
# this function filters the invalid markers
def marker_filter(dataset_name, datapath, count_threshold = 5, single=True):
    
    
    # specify the marker path
    marker_path_extension = ""
    if(single==True):
        marker_path_extension = "single"
    else:
        marker_path_extension = "double"
    # get marker range
    try:
        marker_range = datapath.split("_")[-1].split(".")[0]
        # split start_index and end_index
        start_index = int(marker_range.split("~")[0])
        end_index = int(marker_range.split("~")[1])
        new_marker_range = f"{start_index}~{end_index}"
    except Exception as e:
        marker_range = ""
    output_markerpath = OUTPUT_DATA_DIR + f"{dataset_name}_gennal_filtered_markers_{marker_path_extension}_{new_marker_range}.json"
    
    
    # filter the unvalid markers
    with open(datapath, "r", encoding='utf-8') as f:
        marker_dic = (json.load(f))[0]
    marker_dic_copy = {key: value for key, value in marker_dic.items()}
    for marker in marker_dic_copy:
        if((len(marker) > 30) or (marker_dic_copy[marker][0]["marker_count"] <= count_threshold)):
            marker_dic.pop(marker)
    with open(output_markerpath, "w", encoding='utf-8') as f:
        json.dump([marker_dic], f, ensure_ascii=False, indent=4)
    return output_markerpath




def All_gennal(data, start_index, end_index, dataset_name, filter_threshold):
    responses, gennal_result_path = GenNal(data, start_index, end_index, dataset_name)
    single_marker_path, double_marker_path = marker_counter(dataset_name, None, responses)
    filtered_single_marker_path = marker_filter(dataset_name, single_marker_path, filter_threshold, True)
    filtered_double_marker_path = marker_filter(dataset_name, double_marker_path, filter_threshold, False)
    # marker_graph_single_path = m2g.marker2graph(dataset_name, filtered_single_marker_path, True)
    # marker_graph_double_path = m2g.marker2graph(dataset_name, filtered_double_marker_path, False)
    return filtered_single_marker_path, filtered_double_marker_path, gennal_result_path




# def gennal_marker_processor(markerpath, dataset_name):
#     marker_data = []
#     with open(markerpath, "r", encoding='utf-8') as f:
#         marker_data = json.load(f)
#     single_marker_path, double_marker_path = marker_counter(dataset_name, None, marker_data)
#     filtered_single_marker_path = marker_filter(dataset_name, single_marker_path, FILTER_THRESHOLD, True)
#     filtered_double_marker_path = marker_filter(dataset_name, double_marker_path, FILTER_THRESHOLD, False)
#     marker_graph_single_path = m2g.marker2graph(dataset_name, filtered_single_marker_path, True)
#     marker_graph_double_path = m2g.marker2graph(dataset_name, filtered_double_marker_path, False)
#     return marker_graph_single_path, marker_graph_double_path
