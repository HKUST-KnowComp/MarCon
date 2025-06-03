import json
import os

INSTRUCTION = "You are a helpful assistant with a lot of knowledge."

CSQA_PATH = "path to CSQA dataset"
OUTPUT_DATA_DIR = "output directory for certain (model, dataset)"


# dataset choices
NUM_CHOICES = 4
if(NUM_CHOICES == 4):
    choices = "A, B, C, D"
elif(NUM_CHOICES == 5):
    choices = "A, B, C, D, E"

# Gennal settings
TRAIN_FILTER_THRESHOLD = 0
TEST_FILTER_THRESHOLD = 0
RETRY_NUM = 3
# keyword = "uncertainty" version
GENNAL_PROMPT_SINGLE_MC = f'''The following is a multiple choice question. When responding, answer with a letter from {choices} and incorporate ONLY ONE epistemic marker to reflect your certainty/uncertainty level, DO NOT use numbers. You must include the your choice of letter at the beginning of your response then respond the epistemic markers in a concise yet creative manner. The question is: '''
GENNAL_PROMPT_SINGLE_OPTIONS_MC = f''' and the options are: '''
GENNAL_PROMPT_INDUCE_MC = f''' Please format your answer by surrounding the correct option with two hash symbols on each side, like this: ##X## (X is choosed from {choices}). Your Answer is: '''

# single few shot examples
SINGLE_TEMPLATE = [
{"role": "system", "content": f'''I will provide you a text
with estimation of uncertainty. Please reply ONLY ONE epistemic markers 
from the text to reflect its uncertainty level.
Your should format your output epistemic marker between two pairs of two asterisks. 
Note that {choices} or 'epistemic marker' SHOULD NOT be your output.'''},
{"role": "user", "content": "Based on current knowledge, ethanol production requires more energy."},
{"role": "assistant", "content": "**Based on current knowledge**"},
{"role": "user", "content": "My certainty level in the correctness of the answer is high."},
{"role": "assistant", "content": "**high certainty level**"},
{"role": "user", "content": "I am fairly certain that Harry Potter and the Escape from Gringotts is not a roller coaster ride."},
{"role": "assistant", "content": "**fairly certain**"},
{"role": "user", "content": "Yes, there is a word with 'q' without 'u' for sure."},
{"role": "assistant", "content": "**for sure**"},
{"role": "user", "content": "Yes, the liver is indeed part of the excretory system. \nEpistemic marker: Undoubtedly"},
{"role": "assistant", "content": "**Undoubtedly**"},
{"role": "user", "content": "?\nYes, likely, but I'm not entirely sure, probably. (No)\nYes, it's possible, but I'm fairly certain, actually. (No) \nYes, I'm almost"},
{"role": "assistant", "content": "**not entirely sure**"},
{"role": "user", "content": ", they come from different parts of the pig. So, I am 100%% certain that they are not the same. \nI will use \"certainly\" to signify my certainty level."},
{"role": "assistant", "content": "**certainly**"},
{"role": "user", "content": "?\n\nYes, it was, almost certainly. (1 epistemic marker: 'almost') \nHowever, the correct response format is limited to 10 words, so the revised response would be:\n"},
{"role": "assistant", "content": "**almost certainly**"},
{"role": "user", "content": "?\nYes, they do, probably. (I'm fairly certain about that.)  No, they don't.  Maybe, I'm not entirely sure.  I think so, I'm quite"},
{"role": "assistant", "content": "**fairly certain**"},
]
    
# SINGLE_FEW_SHOT_EXAMPLES = '''
# Here are some examples of correct detection of epistemic markers/words of estimative probability:
# 1.
# <|user|>: Based on current knowledge, ethanol production requires more energy.
# <|assistant|>: **Based on current knowledge**
# 2. 
# <|user|>: "My certainty level in the correctness of the answer is high."
# <|assistant|>: **high certainty level**
# 3.
# <|user|>: "I am fairly certain that Harry Potter and the Escape from Gringotts is not a roller coaster ride."
# <|assistant|>: **fairly certain**
# 4.
# <|user|>: "Yes, there is a word with 'q' without 'u' for sure."
# <|assistant|>: **for sure**
# 5.
# <|user|>: "Yes, Harry Potter and the Escape from Gringotts is a coaster. (High certainty)"
# <|assistant|>: **High certainty**
# 6.
# <|user|>: "Yes, the liver is indeed part of the excretory system. \nEpistemic marker: Undoubtedly"
# <|assistant|>: **Undoubtedly**
# 7.
# <|user|>: "Yes, the Florida Constitution sets a minimum age for legislators. \nEpistemic markers: certainly, definitely, likely, possibly, unlikely, uncertain. I am certainly confident in my response."
# <|assistant|>: **certainly confident**'''

# WEP_DET_PROMPT_SINGLE_1 = f'''I will provide you with a multiple choice question and 
# its corresponding choice with the estimation of uncertainty, 
# which is answered and estimated by another agent.
# Please reply ONLY ONE epistemic markers/words of estimative probability from the agent's
# self-estimation of uncertainty to reflect the uncertainty level. 
# Your should format your output epistemic marker between two 
# pairs of two asterisks. Note that {choices} or 'epistemic marker' SHOULD NOT be your output.
# {SINGLE_FEW_SHOT_EXAMPLES}
# The binary question is: '''
# WEP_DET_PROMPT_SINGLE_2 = ''' And the agent's binary answer with its own estimated uncertainty is:'''
                    
# evaluation hyperparameters
ECE_BINS = 10
MCE_BINS = 10
ACE_BINS = 10