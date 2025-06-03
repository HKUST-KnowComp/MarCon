import os
import transformers
from transformers import AutoModel
from accelerate import init_empty_weights, infer_auto_device_map
import torch
from constants import * 
from transformers import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


model_name = "mistralai/Mistral-7B-Instruct-v0.3"
print("model_name:", model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
model.eval()

def get_response(prompt, instruction, chat_template=None):
    if(chat_template != None):
        inputs = tokenizer.apply_chat_template(chat_template, tokenize=True, 
                                                add_generation_prompt=True, 
                                                return_tensors="pt").to("cuda")
    else:
        inputs = tokenizer.encode(prompt, return_tensors='pt').to("cuda")
    input_length = len(inputs[0])
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length = input_length+30,
            num_return_sequences = 1,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            do_sample = True,
            temperature = 0.5
        )

    generated_text = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    return generated_text

