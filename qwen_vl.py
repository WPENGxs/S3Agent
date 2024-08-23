from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from re_eval import get_predict_label
torch.manual_seed(1234)

import json
from tqdm import tqdm
import re
from qwen_vl_prompt import (
  cot_prompt,
  ape_cot_prompt,
  ps_prompt,
  gkp_prompt_1,
  gkp_prompt_2,
  agent_prompt_0,
  agent_prompt_1,
  agent_prompt_2,
  agent_prompt_3
  )

class qwen_vl():
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda:0", trust_remote_code=True, bf16=True).eval()

  def _get_dataloader(self, json_path):
    data = []
    with open(json_path, 'r') as json_file:
      json_ = json.load(json_file)

    for j in json_:
      image_id = j['image_id']
      text = j['text']
      label = j['label']
      d = (image_id, text, label)
      data.append(d)

    return data
  
  def get_prompt(self, bf_prompt, text):
    return bf_prompt.replace('<text>', text)
  
  def get_json(self, data):
    (image_id, text, label, gen_text, predict_label) = data
    json_ = {
    'image_id': image_id,
    'text': text,
    'label': label,
    'gen_text': str(gen_text),
    'predict_label': predict_label
    }
  
    return json_
  
  def get_json_his(self, data):
    (image_id, text, label, history, predict_label) = data
    json_ = {
    'image_id': image_id,
    'text': text,
    'label': label,
    'gen_text_history': history,
    'predict_label': predict_label
    }
  
    return json_
  
  def qwen_vl_generate_single_text(self, text):
    query = self.tokenizer.from_list_format([
      {'text': text},
    ])
    response, history = self.model.chat(self.tokenizer, query=query, history=None)
    
    return response

  def qwen_vl_generate_single(self, image_path, text):
    query = self.tokenizer.from_list_format([
      {'image': image_path},
      {'text': text},
    ])
    response, history = self.model.chat(self.tokenizer, query=query, history=None)
    
    return response

  def agent(self, image_path_, text):
    history = []
    
    prompt_0 = self.get_prompt(agent_prompt_0, text)
    output_0 = self.qwen_vl_generate_single(image_path_, prompt_0)
    output_0 = str(output_0)
    history.append(output_0)

    prompt_1 = self.get_prompt(agent_prompt_1, text)
    output_1 = self.qwen_vl_generate_single(image_path_, prompt_1)
    output_1 = str(output_1)
    history.append(output_1)

    prompt_2 = self.get_prompt(agent_prompt_2, text)
    output_2 = self.qwen_vl_generate_single(image_path_, prompt_2)
    output_2 = str(output_2)
    history.append(output_2)

    prompt_3 = agent_prompt_3.replace('<output_0>', output_0)
    prompt_3 = prompt_3.replace('<output_1>', output_1)
    prompt_3 = prompt_3.replace('<output_2>', output_2)
    output_3 = self.qwen_vl_generate_single_text(prompt_3)
    output_3 = str(output_3)
    history.append(output_3)

    return history

  def test_agent(self, image_path, json_path):
    data = self._get_dataloader(json_path)

    output_json = []

    true_labels = []
    predict_labels = []

    with tqdm(total=len(data)) as pbar:
      for d in data:
        (image_id, text, label) = d
        image_path_ = f"{image_path}/{image_id}.jpg"
        history = self.agent(image_path_, text)
        answer = history[-1]

        predict_label = get_predict_label(answer)
        if predict_label == -1:
          if label == 0:
            predict_label = 1
          else:
            predict_label = 0

        true_labels.append(label)
        predict_labels.append(predict_label)

        output_data = (image_id, text, label, history, predict_label)
        json_ = self.get_json_his(output_data)
        output_json.append(json_)

        pbar.update(1)
    output_json = json.dumps(output_json)

    output_file = open('./saved_documents/qwen_vl/output_agent.json', 'w', encoding='utf-8')
    output_file.write(output_json)
    output_file.close()

    return (true_labels, predict_labels)