
import torch
from transformers import AutoModel, AutoTokenizer
model_path = '/workspace/models/kobart'
device = 'cuda' if torch.cuda.is_available else 'cpu'

model = AutoModel.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = model.dummy_inputs
attention_mask = inputs['attention_mask']
input_ids = inputs['input_ids']

outputs = model(input_ids, attention_mask, output_hidden_states=True)

print('finished')