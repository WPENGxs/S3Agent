from PIL import Image
from sklearn import metrics
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mmsd2')
parser.add_argument('--model', type=str, default='gemini')
parser.add_argument('--image_path', type=str, default='./image')
parser.add_argument('--text_path', type=str, default='./text')
parser.add_argument('--method', type=str, default='agent')
parser.add_argument('--eval', type=str, default='valid')

def get_eval(true_label, predict_label):
    acc = metrics.accuracy_score(true_label, predict_label)
    f1 = metrics.f1_score(true_label, predict_label, average="binary", pos_label=1)
    precision = metrics.precision_score(true_label, predict_label, average="binary", pos_label=1)
    recall = metrics.recall_score(true_label, predict_label, average="binary", pos_label=1, zero_division=0)
    return (acc, f1, precision, recall)

def check_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"An error occurred while creating path '{path}': {e.strerror}")

def main():
    args = parser.parse_args()
    image_path = args.image_path

    if args.dataset == 'mmsd2':
        json_path = f'{args.text_path}/text_json_final/{args.eval}.json'
    else:
        json_path = f'{args.text_path}/text_json_clean/{args.eval}.json'

    check_dir(f'./saved_documents/{args.model}')

    if args.model == 'gemini_pro':
        from gemini_pro import gemini_pro
        g = gemini_pro()
    elif args.model == 'qwen_vl':
        from qwen_vl import qwen_vl
        g = qwen_vl()
    elif args.model == 'yi_vl':
        from yi_vl import yi_vl
        g = yi_vl()
    elif args.model == 'minicpm_v2':
        from minicpm_v2 import minicpm_v2
        g = minicpm_v2()
    elif args.model == 'llava_v1_5':
        from llava_v1_5 import llava_v1_5
        g = llava_v1_5()
    elif args.model == 'deepseek_vl_chat':
        from deepseek_vl_chat import deepseek_vl_chat
        g = deepseek_vl_chat()

    if args.method == 'agent':
        (true_labels, predict_labels) = g.test_agent(image_path, json_path)
    else:
        print('no method')

    (acc, f1, precision, recall) = get_eval(true_labels, predict_labels)
    print('#' * 80)
    print('acc:', acc, ' f1:', f1, ' precision:', precision, ' recall:', recall)
    print('#' * 80)

if __name__ == '__main__':
    main()
