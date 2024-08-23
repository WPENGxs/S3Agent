import google.generativeai as genai
from PIL import Image
import json
from tqdm import tqdm
import re
import time
import requests
from re_eval import get_predict_label
from gemini_pro_prompt import (
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

# Gemini api
# document: https://ai.google.dev/api/python/google/generativeai

genai.configure(api_key='your api key')
config = genai.types.GenerationConfig(
        candidate_count=1,
        stop_sequences=None,
        max_output_tokens=2048,
        temperature=0.9,
        top_k=1,
        top_p=1
        )
safety_settings = [
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]

class gemini_pro():
  def __init__(self):
    self.model = genai.GenerativeModel(model_name='gemini-pro-vision',
                                       safety_settings=safety_settings)
    self.model_text = genai.GenerativeModel(model_name='gemini-pro',
                                        safety_settings=safety_settings)

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
  
  def gemini_generate_single_text(self, text):
    message = ''
    while True:
      try:
        response = self.model_text.generate_content(text, stream=False, generation_config=config)
        response.resolve()
        try:
          message = response.text
        except:
          message = 'Error'
      except:
        message = 'Error'
      if message != 'Error':
        break
    return message

  def gemini_generate_single(self, image, text):
    message = ''
    while True:
      try:
        response = self.model.generate_content([text, image], stream=False, 
                                              generation_config=config)
        response.resolve()
        try:
          message = response.text
        except:
          message = 'Error'
      except:
        message = 'Error'
      if message != 'Error':
        break
    return message

  def agent(self, image, text):
    history = []
    
    prompt_0 = self.get_prompt(agent_prompt_0, text)
    output_0 = self.gemini_generate_single(image, prompt_0)
    output_0 = str(output_0)
    history.append(output_0)

    prompt_1 = self.get_prompt(agent_prompt_1, text)
    output_1 = self.gemini_generate_single(image, prompt_1)
    output_1 = str(output_1)
    history.append(output_1)

    prompt_2 = self.get_prompt(agent_prompt_2, text)
    output_2 = self.gemini_generate_single(image, prompt_2)
    output_2 = str(output_2)
    history.append(output_2)

    prompt_3 = agent_prompt_3.replace('<output_0>', output_0)
    prompt_3 = prompt_3.replace('<output_1>', output_1)
    prompt_3 = prompt_3.replace('<output_2>', output_2)
    output_3 = self.gemini_generate_single_text(prompt_3)
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
        image = Image.open(f"{image_path}/{image_id}.jpg")

        history = self.agent(image, text)
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

    output_file = open('./saved_documents/gemini_pro/output_agent.json', 'w', encoding='utf-8')
    output_file.write(output_json)
    output_file.close()

    return (true_labels, predict_labels)