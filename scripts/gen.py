import yaml
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def batch_loader(seq, size):
    """
    Iterator that takes in a list `seq` and returns
    chunks of size `size` """
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def get_prompt_dataset(path_to_prompts):
    with open(prompts) as f:
        src_li = f.readlines()

    # change this line if want different size
    src_df = pd.DataFrame(src_li[0:5000])
    test_desc = '<|startoftext|>' + src_df[0] + '[seperator]'
    return test_desc


# get config
with open('configs/pythia_config.yml', 'r') as stream:
    cfg = yaml.safe_load(stream)['val']

tokenizer_name = cfg['tokenizer']
model_name = cfg['model']
prompts = cfg['prompts']
n = cfg['n']
batch_size = cfg['batch_size']
output_path = cfg['output']
max_length = cfg['max_length']
max_new_tokens = cfg['max_new_tokens']


# are we sampling output
sampling = False
if n > 1:
    sampling = True


# get tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, 
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>', 
                                          padding_side='left',
                                          truncation_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name).eval().cuda()
model.resize_token_embeddings(len(tokenizer))

test_desc = get_prompt_dataset(prompts)

print("Generating outputs...\n")
output = []
batched_dataset = batch_loader(list(test_desc), batch_size)  
for batch in tqdm(batched_dataset):
    tokenized_inputs = tokenizer(batch, return_tensors="pt", max_length=max_length, padding=True, truncation=True)
    tokenized_inputs = {x: tokenized_inputs[x].cuda() for x in tokenized_inputs}
    # need to add following line when dealing with neoX models
    tokenized_inputs.pop('token_type_ids')
    tokenized_outputs = model.generate(**tokenized_inputs, 
                                       max_new_tokens=max_new_tokens, 
                                       do_sample=sampling, 
                                       num_return_sequences=n,
                                       bos_token_id=tokenizer.bos_token_id, 
                                       eos_token_id=tokenizer.eos_token_id, 
                                       pad_token_id=tokenizer.pad_token_id)
    print(tokenized_outputs)
    for out in tokenized_outputs:
        output.append(tokenizer.decode(out, skip_special_tokens=True).replace("\n", "")+"\n")


# write output
with open(output_path, 'w') as f:
    f.writelines(output)

print("Model output saved to " + output_path)
