import torch.nn as nn
from transformers import BartForConditionalGeneration, AutoTokenizer
from sklearn.metrics import accuracy_score


class MultitaskDeletionAbsSummaryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False, use_fast=False)
        
        self.hidden_size = self.model.config.hidden_size
        self.classification_layer = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, abs_labels=None):
        outputs = self.model(input_ids,
                            attention_mask=attention_mask, 
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=abs_labels)
        encoder_hidden_states = outputs.encoder_last_hidden_state
        encoder_classified_tokens = self.classification_layer(encoder_hidden_states)
        encoder_classified_tokens = self.sigmoid(encoder_classified_tokens)
        return outputs, encoder_classified_tokens


if __name__ == '__main__':
    # Model test
    import argparse
    from utils import *
    from data_utils import *
    import torch
    from rouge import Rouge
    import torch.nn as nn
    softmax = nn.Softmax()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'models/kobart'

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='kobart.json')
    parser.add_argument("--output_dir", type=str, default='kobart-ckpt')
    args = parser.parse_args()
    args = init_setting(args)

    model = MultitaskDeletionAbsSummaryModel(args).to(device)
    criterion = nn.BCELoss()

    tokenizer = model.tokenizer
    model_path = 'models/kobart'
    
    data_loader = get_dataloader(args, tokenizer, 'train')
    rouge_scorer = Rouge()
    for batch in data_loader:
        input_ids, attention_mask, decoder_input_ids, ext_labels, abs_labels, content, target_texts = batch
        input_ids, attention_mask, decoder_input_ids, ext_labels, abs_labels = \
            input_ids.to(device), attention_mask.to(device), decoder_input_ids.to(device), ext_labels.to(device), abs_labels.to(device)
        decoder_attention_mask = decoder_input_ids.ne(3).float().to(device)

        outputs, encoder_classified_token = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, abs_labels)
        
        decode_loss = outputs.loss
        encode_loss = criterion(encoder_classified_token.squeeze(-1), ext_labels)

        loss = encode_loss + decode_loss
        # loss.backward()

        # Encoder scoring
        preds = (encoder_classified_token > 0.5)
        flatten_preds = torch.flatten(preds).tolist()
        flatten_labels = torch.flatten(ext_labels).tolist()
        encoder_acc_score = accuracy_score(flatten_preds, flatten_labels)

        # Decoder scoring
        decoded_label = [' '.join(tokenizer.batch_decode(label, skip_special_tokens=True)) for label in abs_labels]
        tokenized_label = [' '.join(tokenizer.tokenize(label)) for label in decoded_label]
        
        decoded_output = tokenizer.batch_decode(torch.topk(softmax(outputs.logits), 1, -1).indices.squeeze(-1), skip_special_tokens=True)
        tokenized_output = [' '.join(tokenizer.tokenize(output_text)) for output_text in decoded_output]
        
        scores = rouge_scorer.get_scores(tokenized_output, tokenized_label, avg=True)
        rouge1 = scores['rouge-1']['f']
        rouge2 = scores['rouge-2']['f']
        rougel = scores['rouge-l']['f']

        
