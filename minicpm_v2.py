import torch
torch.manual_seed(1234)
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from re_eval import get_predict_label

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
import json
from tqdm import tqdm
import re
from minicpm_v2_prompt import (
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

class minicpm_v2():
  def __init__(self):
    disable_torch_init()
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)
    self.model = model.to(device='cuda:0', dtype=torch.bfloat16)

    self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
    self.model.eval()

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
  
  def minicpm_v2_generate_single_text(self, text):
    image = Image.open('./img/blank_512.jpg').convert('RGB')
    question = text
    msgs = [{'role': 'user', 'content': question}]

    res, context, _ = self.model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=self.tokenizer,
        sampling=True,
        temperature=0.7
    )
    return res

  def minicpm_v2_generate_single(self, image_path, text):
    image = Image.open(image_path).convert('RGB')
    question = text
    msgs = [{'role': 'user', 'content': question}]

    res, context, _ = self.model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=self.tokenizer,
        sampling=True,
        temperature=0.7
    )
    return res

  def agent(self, image_path_, text):
    history = []
    
    prompt_0 = self.get_prompt(agent_prompt_0, text)
    output_0 = self.minicpm_v2_generate_single(image_path_, prompt_0)
    output_0 = str(output_0)
    history.append(output_0)

    prompt_1 = self.get_prompt(agent_prompt_1, text)
    output_1 = self.minicpm_v2_generate_single(image_path_, prompt_1)
    output_1 = str(output_1)
    history.append(output_1)

    prompt_2 = self.get_prompt(agent_prompt_2, text)
    output_2 = self.minicpm_v2_generate_single(image_path_, prompt_2)
    output_2 = str(output_2)
    history.append(output_2)

    prompt_3 = agent_prompt_3.replace('<output_0>', output_0)
    prompt_3 = prompt_3.replace('<output_1>', output_1)
    prompt_3 = prompt_3.replace('<output_2>', output_2)
    output_3 = self.minicpm_v2_generate_single_text(prompt_3)
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

    output_file = open('./saved_documents/minicpm_v2/output_agent.json', 'w', encoding='utf-8')
    output_file.write(output_json)
    output_file.close()

    return (true_labels, predict_labels)