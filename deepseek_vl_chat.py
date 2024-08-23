import torch
torch.manual_seed(1234)
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from re_eval import get_predict_label

import json
from tqdm import tqdm
import re
from deepseek_vl_prompt import (
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

class deepseek_vl_chat():
  def __init__(self):
    model_path = "./deepseek-vl-7b-chat"
    self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    self.tokenizer = self.vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda(0).eval()

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
  
  def deepseek_vl_generate_single_text(self, text):
    conversation = [
    {
        "role": "User",
        "content": text,
    },
    {
        "role": "Assistant",
        "content": ""
    }
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = self.vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
    ).to(self.vl_gpt.device)

    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = self.vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=self.tokenizer.eos_token_id,
    bos_token_id=self.tokenizer.bos_token_id,
    eos_token_id=self.tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
    )

    answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

  def deepseek_vl_generate_single(self, image_path, text):
    
    conversation = [
    {
        "role": "User",
        "content": text,
        "images": [image_path]
    },
    {
        "role": "Assistant",
        "content": ""
    }
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = self.vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
    ).to(self.vl_gpt.device)

    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = self.vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=self.tokenizer.eos_token_id,
    bos_token_id=self.tokenizer.bos_token_id,
    eos_token_id=self.tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
    )

    answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

  def agent(self, image_path_, text):
    history = []
    
    prompt_0 = self.get_prompt(agent_prompt_0, text)
    output_0 = self.deepseek_vl_generate_single(image_path_, prompt_0)
    output_0 = str(output_0)
    history.append(output_0)

    prompt_1 = self.get_prompt(agent_prompt_1, text)
    output_1 = self.deepseek_vl_generate_single(image_path_, prompt_1)
    output_1 = str(output_1)
    history.append(output_1)

    prompt_2 = self.get_prompt(agent_prompt_2, text)
    output_2 = self.deepseek_vl_generate_single(image_path_, prompt_2)
    output_2 = str(output_2)
    history.append(output_2)

    prompt_3 = agent_prompt_3.replace('<output_0>', output_0)
    prompt_3 = prompt_3.replace('<output_1>', output_1)
    prompt_3 = prompt_3.replace('<output_2>', output_2)
    output_3 = self.deepseek_vl_generate_single_text(prompt_3)
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

    output_file = open('./saved_documents/deepseek_vl_chat/output_agent.json', 'w', encoding='utf-8')
    output_file.write(output_json)
    output_file.close()

    return (true_labels, predict_labels)
