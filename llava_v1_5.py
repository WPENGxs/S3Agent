import torch
torch.manual_seed(1234)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
torch.manual_seed(1234)
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
from llava_v1_5_prompt import (
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

class llava_v1_5():
  def __init__(self):
    disable_torch_init()
    self.model_path = './llava-v1.5-13b'
    self.model_base = None
    self.load_8bit = False
    self.load_4bit = False
    self.device_map = 'cuda:0'
    self.device = 'cuda'
    self.temperature = 0.2
    self.max_new_tokens = 512
    
    conv_mode = None

    self.model_name = get_model_name_from_path(self.model_path)
    self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name, self.load_8bit, self.load_4bit, device_map=self.device_map, device=self.device)

    if "llama-2" in self.model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in self.model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in self.model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in self.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in self.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if conv_mode is not None:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, conv_mode, conv_mode))
    else:
        conv_mode = conv_mode
    self.conv_mode = conv_mode

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
  
  def llava_v1_5_generate_single_text(self, text):
    conv = conv_templates[self.conv_mode].copy()
    inp = text
    image = Image.open('./img/blank_512.jpg').convert('RGB')
    image_size = image.size
    
    image_tensor = process_images([image], self.image_processor, self.model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

    if image is not None:
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    with torch.inference_mode():
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            use_cache=True)

    outputs = self.tokenizer.decode(output_ids[0]).strip()
    return outputs.replace('<s>', '').replace('</s>', '')

  def llava_v1_5_generate_single(self, image_path, text):
    conv = conv_templates[self.conv_mode].copy()
    inp = text
    image = Image.open(image_path).convert('RGB')
    image_size = image.size
    
    image_tensor = process_images([image], self.image_processor, self.model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

    if image is not None:
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    with torch.inference_mode():
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            use_cache=True)

    outputs = self.tokenizer.decode(output_ids[0]).strip()

    return outputs.replace('<s>', '').replace('</s>', '')

  def agent(self, image_path_, text):
    history = []
    
    prompt_0 = self.get_prompt(agent_prompt_0, text)
    output_0 = self.llava_v1_5_generate_single(image_path_, prompt_0)
    output_0 = str(output_0)
    history.append(output_0)

    prompt_1 = self.get_prompt(agent_prompt_1, text)
    output_1 = self.llava_v1_5_generate_single(image_path_, prompt_1)
    output_1 = str(output_1)
    history.append(output_1)

    prompt_2 = self.get_prompt(agent_prompt_2, text)
    output_2 = self.llava_v1_5_generate_single(image_path_, prompt_2)
    output_2 = str(output_2)
    history.append(output_2)

    prompt_3 = agent_prompt_3.replace('<output_0>', output_0)
    prompt_3 = prompt_3.replace('<output_1>', output_1)
    prompt_3 = prompt_3.replace('<output_2>', output_2)
    output_3 = self.llava_v1_5_generate_single_text(prompt_3)
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

    output_file = open('./saved_documents/llava_v1_5/output_agent.json', 'w', encoding='utf-8')
    output_file.write(output_json)
    output_file.close()

    return (true_labels, predict_labels)