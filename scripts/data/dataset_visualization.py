import pickle
import sys
import math
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


"""
This script visualizes the dataset.
It takes an argument that is path to the folder containing the source.txt and target.txt to visualize.
It outputs graphs and also dumps the length data into pickles.
"""


def batch_loader(seq, size):
    return [seq[pos:pos+size] for pos in range(0, len(seq), size)]
    
def get_plot(inpath, outpath, tokenizer, n, name):
    with open(inpath, 'r') as f:
        li = f.readlines()
    tokenized = []
    for batch in tqdm(batch_loader(li, 2048)):
        tokenized += tokenizer(batch)['input_ids']
    len_li = [len(x) for x in tokenized]
    plt.figure(n)
    n, bins, patches = plt.hist(len_li, 20, log=True)
    plt.title(name)
    plt.xlabel('Tokenized Length')
    plt.ylabel('Number of Examples')
    plt.savefig(outpath)
    print("{} has n={} and bins={}".format(name,n,bins))
    return len_li

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-125m-deduped')
    
train_src = get_plot(sys.argv[1] + 'source.txt', 'train_source.png', tokenizer, 1, 'Training Dataset Source')
train_tgt = get_plot(sys.argv[1] + 'target.txt', 'train_target.png', tokenizer, 2, 'Training Dataset Target')

total_train = [x[0] + x[1] + 1 for x in zip(train_src, train_tgt)]
plt.figure(3)
plt.hist(total_train, 20, log=True)
plt.title('Training Dataset')
plt.xlabel('Tokenized Length')
plt.ylabel('Number of Examples')
plt.savefig('total_train.png')

with open('source_len.txt', 'wb') as f:
    pickle.dump(train_src, f)
with open('target_len.txt', 'wb') as f:
    pickle.dump(train_tgt, f)
with open('total_len.txt', 'wb') as f:
    pickle.dump(total_train, f)
    