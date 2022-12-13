import os
import yaml
import math
import pandas as pd

import torch
import torch.nn
from torch.utils.data import Dataset, random_split

import transformers
from transformers import Trainer, IntervalStrategy, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names


class IsarStep(Dataset):
    def __init__(self, src_li, tgt_li, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.source = src_li
        self.target = tgt_li
        #   for txt in txt_list:
        #       encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
        #                                  max_length=max_length, padding="max_length")
        #       self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
        #       self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        full_text = '<|startoftext|>' + self.source[idx] + '[seperator]' + self.target[idx] + '<|endoftext|>' 
        full_text_encode = self.tokenizer(full_text,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.max_length,
                                          return_tensors='pt')
        ids = full_text_encode['input_ids'].squeeze()
        attn_masks = full_text_encode['attention_mask'].squeeze()
        return ids.long(), attn_masks.long()


def load_dataset(path, tokenizer, max_length):

    with open(path + 'source.txt') as f:
      src_li = f.readlines()
    
    with open(path + 'target.txt') as f:
      tgt_li = f.readlines()
    
    # descriptions = ['<|startoftext|>' + x + '[seperator]' + y + '<|startoftext|>' for (x,y) in zip(src_df, tgt_df)]
    # src_df = pd.DataFrame(src_li)
    # tgt_df = pd.DataFrame(tgt_li)
    # 
    # src_df['text'] = src_df[0] + '[seperator]' + tgt_df[0]
    # descriptions = src_df['text']
    dataset = IsarStep(src_li, tgt_li, tokenizer, max_length=max_length)
    return dataset


def gptneo_data_collator(data):
    return {
            'input_ids': torch.stack([f[0] for f in data]), 
            'attention_mask': torch.stack([f[1] for f in data]),
            'labels': torch.stack([f[0] for f in data])
            }


# get config
with open('./configs/pythia_config.yml', 'r') as stream:
    cfg = yaml.safe_load(stream)['train']

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["devices"]
tokenizer_name=cfg["tokenizer"]
model_name=cfg["model"]
data_path=cfg["data_path"]
batch_size=cfg["batch_size"]
grad_accum=cfg["grad_accum"]
weight_decay=cfg["weight_decay"]
lr=cfg["lr"]
epochs=cfg["epochs"]
model_output_path=cfg["model_output_path"]
max_length=cfg["max_length"]


# get model and tokenizer
print('Downloading model and tokenizer...\n')
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>', 
                                          truncation_side='left', 
                                          padding_side='left')
model.resize_token_embeddings(len(tokenizer))


# load and split dataset
print('Loading dataset...')
dataset = load_dataset(data_path, tokenizer, max_length)
print(len(dataset))
train_size = int(0.95 * len(dataset))
print('Splitting dataset...')
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
# train_dataset, val_dataset = load_dataset(data_path, tokenizer, max_length)
# print("Training dataset length: " + str(len(train_dataset)))
# print("Validation dataset length: " + str(len(val_dataset)))


# Optimizer 
# decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
# decay_parameters = [name for name in decay_parameters if "bias" not in name]
# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if n in decay_parameters],
#         "weight_decay": weight_decay,
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
#         "weight_decay": 0.0,
#     },
# ]

num_gpus = torch.cuda.device_count()
steps_per_epoch = math.floor(len(dataset)/(batch_size*num_gpus*grad_accum))
print("GPUs = {}".format(num_gpus))
print("Steps per epoch = {}".format(steps_per_epoch))
# optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
# scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
#                                                          100, 
#                                                          epochs*steps_per_epoch, 
#                                                          )

# training step
print('Training...\n')
training_args = TrainingArguments(output_dir='./results', 
                                  num_train_epochs=epochs,
                                  # logging_steps=steps_per_epoch*10, 
                                  fp16=True,
                                  per_device_train_batch_size=batch_size, 
                                  per_device_eval_batch_size=batch_size,
                                  warmup_steps=100, 
                                  weight_decay=0.01, 
                                  # save_steps=steps_per_epoch*10,
                                  logging_dir='./logs',
                                  logging_steps=math.floor(steps_per_epoch/5),
                                  # max_grad_norm=1.0,
                                  gradient_accumulation_steps=grad_accum,
                                  evaluation_strategy='steps',
                                  save_strategy='steps',
                                  eval_steps=math.floor(steps_per_epoch/5),
                                  save_steps=math.floor(steps_per_epoch/5))
Trainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        data_collator=gptneo_data_collator).train()


# save model and tokenizer
print('Saving models...\n')
model.save_pretrained(model_output_path)
