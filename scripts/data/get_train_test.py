import sys
import pandas as pd


with open(sys.argv[1], 'r') as f:
    src_li = f.readlines()

with open(sys.argv[2], 'r') as f:
    tgt_li = f.readlines()

src_df = pd.DataFrame(src_li)
tgt_df = pd.DataFrame(tgt_li)

df = src_df[0].str.strip() + '[seperator]' + tgt_df[0].str.strip()

train = df.sample(frac=0.9, random_state=200)
val = df.drop(train.index).sample(frac=1.0)

train.to_csv('train_set', index=False)
val.to_csv('val_set', index=False)
