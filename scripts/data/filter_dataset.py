import sys
import math
from tqdm import tqdm
from transformers import AutoTokenizer

"""
This script takes a source file and target file and outputs a file 
with entries <source> + [seperator] + <target> that has sequence length
less than n.
"""

def batch_loader(seq, size):
    return [seq[pos:pos+size] for pos in range(0, len(seq), size)]

n = 512

print('Opening files...')
with open(sys.argv[1], 'r') as f:
    src_li = f.read().splitlines()
    
with open(sys.argv[2], 'r') as f:
    tgt_li = f.read().splitlines()
    
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-125m-deduped',
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>', 
                                          truncation_side='left', 
                                          padding_side='left')

print('Constructing concat list...')
concat = [x + '[seperator]' + y + '\n' for x,y in zip(src_li, tgt_li)]

output = []
print('Tokenizing...')
for batch in tqdm(batch_loader(concat, 2048)):
    tokenized = tokenizer(batch)['input_ids']
    temp = [x for i,x in enumerate(batch) if len(tokenized[i]) < n]
    output += temp
    
# print('Filtering...')
# output = [concat[i + 2*math.floor(len(concat)/3)] for i,x in enumerate(tokenized) if len(x) < n]
with open('512_dataset_train.txt', 'w') as f:
    f.writelines(output)

print('Dataset written to 512_dataset_train.txt')