import json
from sklearn import metrics
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='all')
parser.add_argument('--method', type=str, default='all')

def get_eval(true_label, predict_label):
    acc = metrics.accuracy_score(true_label, predict_label)
    f1 = metrics.f1_score(true_label, predict_label, average="binary", pos_label=1)
    precision = metrics.precision_score(true_label, predict_label, average="binary", pos_label=1)
    recall = metrics.recall_score(true_label, predict_label, average="binary", pos_label=1, zero_division=0)
    return (acc, f1, precision, recall)

def get_predict_label(gen_text):
    predict_label = -1

    gen_text = str(gen_text)
    answer = re.search(r'(?<="Answer": ")[^"]+', gen_text)
    if answer:
        answer = answer.group()
        if answer == 'yes' or answer == '<<<yes>>>' or answer == 'Yes':
            predict_label = 1
        elif answer == 'no' or answer == '<<<no>>>' or answer == 'No':
            predict_label = 0
        else:
            predict_label = -1
    answer = re.search(r'Answer: (yes|no)', gen_text)
    if answer:
        answer = answer.group(1)
        if answer == 'yes':
            predict_label = 1
        elif answer == 'no':
            predict_label = 0
        else:
            predict_label = -1
    answer = re.search(r'Answer: (Yes|No)', gen_text)
    if answer:
        answer = answer.group(1)
        if answer == 'Yes':
            predict_label = 1
        elif answer == 'No':
            predict_label = 0
        else:
            predict_label = -1
    answer = re.search(r'Answer: (<<<yes>>>|<<<no>>>)', gen_text)
    if answer:
        answer = answer.group(1)
        if answer == '<<<yes>>>':
            predict_label = 1
        elif answer == '<<<no>>>':
            predict_label = 0
        else:
            predict_label = -1
    answer = re.search(r'(<yes>|<no>)', gen_text)
    if answer:
        answer = answer.group(1)
        if answer == '<yes>':
            predict_label = 1
        elif answer == '<no>':
            predict_label = 0
        else:
            predict_label = -1

    return predict_label

def output_eval_method(model, method):
    print('#' * 80)
    print(model)
    print('#' * 80)
    has_history = False
    if method == 'agent':
        has_history = True

    json_path = f'./saved_documents/{model}/output_{method}.json'

    with open(json_path, 'r') as json_file:
        json_ = json.load(json_file)
    
    true_labels = []
    predict_labels = []

    p_true_labels = []
    p_predict_labels = []

    n_true_labels = []
    n_predict_labels = []

    for j in json_:
        label = j['label']
        true_labels.append(label)

        if has_history:
            gen_text = j['gen_text_history'][-1]
        else:
            gen_text = j['gen_text']
        predict_label = get_predict_label(gen_text)
        if predict_label == -1:
            if label == 0:
                predict_label = 1
            else:
                predict_label = 0

        if label == 1:
            p_true_labels.append(label)
            p_predict_labels.append(predict_label)
        elif label == 0:
            n_true_labels.append(label)
            n_predict_labels.append(predict_label)

        predict_labels.append(predict_label)

        j['predict_label'] = predict_label
        
    (acc, f1, precision, recall) = get_eval(true_labels, predict_labels)
    (p_acc, p_f1, p_precision, p_recall) = get_eval(p_true_labels, p_predict_labels)
    (n_acc, n_f1, n_precision, n_recall) = get_eval(n_true_labels, n_predict_labels)

    print(method)
    print('acc:', acc, ' f1:', f1, ' precision:', precision, ' recall:', recall)
    print('#' * 80)

def output_eval_all(model):
    print('#' * 80)
    print(model)
    print('#' * 80)
    methods = ['cot', 'ape_cot', 'ps', 'gkp', 'agent']
    for method in methods:
        has_history = False
        if method == 'agent':
            has_history = True

        json_path = f'./saved_documents/{model}/output_{method}.json'

        with open(json_path, 'r') as json_file:
            json_ = json.load(json_file)
        
        true_labels = []
        predict_labels = []

        p_true_labels = []
        p_predict_labels = []

        n_true_labels = []
        n_predict_labels = []

        for j in json_:
            label = j['label']
            true_labels.append(label)

            if has_history:
                gen_text = j['gen_text_history'][-1]
            else:
                gen_text = j['gen_text']
            predict_label = get_predict_label(gen_text)
            if predict_label == -1:
                if label == 0:
                    predict_label = 1
                else:
                    predict_label = 0

            if label == 1:
                p_true_labels.append(label)
                p_predict_labels.append(predict_label)
            elif label == 0:
                n_true_labels.append(label)
                n_predict_labels.append(predict_label)

            predict_labels.append(predict_label)

            j['predict_label'] = predict_label
        
        (acc, f1, precision, recall) = get_eval(true_labels, predict_labels)
        (p_acc, p_f1, p_precision, p_recall) = get_eval(p_true_labels, p_predict_labels)
        (n_acc, n_f1, n_precision, n_recall) = get_eval(n_true_labels, n_predict_labels)

        print(method)
        print('acc:', acc, ' f1:', f1, ' precision:', precision, ' recall:', recall)
        print('#' * 80)

if __name__ == '__main__':
    models = ['gemini_pro', 'qwen_vl', 'yi_vl', 'minicpm_v2', 'llava_v1_5_7b', 'llava_v1_5_13b', 'deepseek_vl_chat']
    methods = ['cot', 'ape_cot', 'ps', 'gkp', 'agent']
    args = parser.parse_args()
    if args.model != 'all':
        model = args.model
        if args.method != 'all':
            method = args.method
            output_eval_method(model, method)
        else:
            output_eval_all(model)
    else:
        if args.method != 'all':
            method = args.method
            for model in models:
                output_eval_method(model, method)
        else:
            for model in models:
                output_eval_all(model)