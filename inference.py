import sys, os
import argparse
sys.path.append('src')
import torch
from summary_model import DeletionBasedSummaryModel
from data_utils import get_dataloader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(args):
    model_name = 'intent_class_model.pt'
    model_path = 'koelectra_small_ckpt/checkpoint-1500'
    model = torch.load(os.path.join(model_path, model_name))
    tokenizer = model.tokenizer    

    test_dataloader = get_dataloader(args, tokenizer, data_type='test')

    target_list = []
    infer_list = []
    for batch in tqdm(test_dataloader):
        inputs, labels, target_text = batch
        inputs, labels = inputs.to(device), labels.to(device)
        input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']

        outputs = model(input_ids, attention_mask, token_type_ids)
        preds = (outputs.squeeze(-1) > 0.5)

        preds_ids = (preds*input_ids).tolist()
        preds_tokens = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)

        target_list += target_text
        infer_list += preds_tokens
        # tokenizer.batch_decode((labels.type(torch.long)*input_ids).tolist(), skip_special_tokens=True)
        
        print(1)
        


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", type=str, default='intent_class_model.pt')
    arg_parser.add_argument("--model_path", type=str, default='koelectra_small_ckpt/checkpoint-7500')
    arg_parser.add_argument("--data_dir", type=str, default='data/opinion')
    arg_parser.add_argument("--valid_file", type=str, default='valid_dataset.json')
    arg_parser.add_argument("--max_seq_length", type=int, default=512)
    arg_parser.add_argument("--valid_batch_size", type=int, default=16)
    args = arg_parser.parse_args()
    inference(args)