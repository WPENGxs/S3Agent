import os
import torch
from llava_yi.conversation import conv_templates
from llava_yi.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava_yi.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
from re_eval import get_predict_label

CUDA_VISIBLE_DEVICES=1

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

import torch
torch.manual_seed(1234)

import json
from tqdm import tqdm
import re
from yi_vl_prompt import (
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

class yi_vl():
  def __init__(self):
    disable_torch_init()
    model_path = os.path.expanduser("01-ai/Yi-VL-6B")
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path)

    self.temperature=0.8
    self.top_p=1.0
    self.top_k=40
    self.num_beams=1

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
  
  def yi_vl_generate_single_text(self, text):
    image_file = './img/blank_512.jpg'
    qs = text
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["mm_default"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    image = Image.open(image_file)
    if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in self.image_processor.image_mean)
        )
    image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
    model = self.model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=1024,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = self.tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    
    return outputs

  def yi_vl_generate_single(self, image_path, text):
    image_file = image_path
    qs = text
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["mm_default"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    image = Image.open(image_file)
    if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in self.image_processor.image_mean)
        )
    image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
    model = self.model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=1024,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = self.tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    
    return outputs

  def agent(self, image_path_, text):
    history = []
    
    prompt_0 = self.get_prompt(agent_prompt_0, text)
    output_0 = self.yi_vl_generate_single(image_path_, prompt_0)
    output_0 = str(output_0)
    history.append(output_0)

    prompt_1 = self.get_prompt(agent_prompt_1, text)
    output_1 = self.yi_vl_generate_single(image_path_, prompt_1)
    output_1 = str(output_1)
    history.append(output_1)

    prompt_2 = self.get_prompt(agent_prompt_2, text)
    output_2 = self.yi_vl_generate_single(image_path_, prompt_2)
    output_2 = str(output_2)
    history.append(output_2)

    prompt_3 = agent_prompt_3.replace('<output_0>', output_0)
    prompt_3 = prompt_3.replace('<output_1>', output_1)
    prompt_3 = prompt_3.replace('<output_2>', output_2)
    output_3 = self.yi_vl_generate_single_text(prompt_3)
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

    output_file = open('./saved_documents/yi_vl/output_agent.json', 'w', encoding='utf-8')
    output_file.write(output_json)
    output_file.close()

    return (true_labels, predict_labels)