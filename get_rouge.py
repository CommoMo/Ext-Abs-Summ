import torch
from tqdm import tqdm
import json, os, sys
from rouge import Rouge
from transformers import AutoTokenizer
from nltk import sent_tokenize
import argparse

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=125000, type=int)
    args = parser.parse_args()
    return args

args = get_args()
ckpt = args.ckpt

sys.path.append('src')  

device = 'cuda'
ckpt_list = [_.split('-')[-1] for _ in os.listdir('/workspace/ckpt/kobart_ckpt') if _.startswith('checkpoint')]
ckpt_list.sort()

test_dataset = json.load(open('data/article/valid_dataset.json', 'r', encoding='utf-8'))
tokenizer = AutoTokenizer.from_pretrained('models/kobart')
rouge_scorer = Rouge()
target_list = [target['abs'] for target in test_dataset]

def tokenize_list(data_list):
    return [' '.join(tokenizer.tokenize(data)) for data in data_list]


tokenized_target_list = tokenize_list(target_list)
best_rouge = 0

# for ckpt in ckpt_list:
result_list = []
model_path = f'/workspace/ckpt/kobart_ckpt/checkpoint-{ckpt}'
model = torch.load(os.path.join(model_path, 'multitask_ext_abs_summary_model.pt')).model.to(device)
print(f"Current doing... {model_path.split('/')[-1]}")

for data in tqdm(test_dataset):
    doc = ' '.join(data['sentences'])
    input_ids = tokenizer(doc, return_tensors="pt").input_ids.to(device)

    output = model.generate(input_ids, num_beams=5, eos_token_id=1, repetition_penalty=1.2, no_repeat_ngram_size=1, early_stopping=True,
                            max_length=150)
    result = tokenizer.decode(output[0], skip_special_toknes=True).replace('</s>', '')
    result = sent_tokenize(result)[0]
    result_list.append(result)

tokenized_result_list = tokenize_list(result_list)
scores = rouge_scorer.get_scores(tokenized_result_list, tokenized_target_list, avg=True)
rouge_score = scores['rouge-l']['f']
if rouge_score > best_rouge:
    best_rouge = rouge_score
    best_score = scores
    best_ckpt = ckpt
print(f"Best CKPT: {best_ckpt}, Best score: {best_score}")

with open(f'results/infer/{ckpt}_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(result_list))

with open(f'results/rouge/{ckpt}_rouge.json', 'w') as f:
    json.dump(best_score, f, indent='\t')