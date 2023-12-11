from datasets import load_dataset, Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import deepspeed
import os
from tqdm import tqdm
from transformers import pipeline
import pandas as pd

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))


test = pd.read_csv("test_formatted.csv")
print(test.head())
test_dataset = Dataset.from_pandas(test)



model_id = "codellama/CodeLlama-7b-Instruct-hf"


generator = pipeline('text-generation', 
                     model=model_id,
                     device_map=local_rank,
                     max_new_tokens=256)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float16,
                                           max_out_tokens=2048,
                                           replace_with_kernel_inject=True)
generator.tokenizer.padding_side='left'



results = []

for i in tqdm(range(len(test_dataset['baseline_non_instruct']))):
    data_ex = test_dataset['baseline_non_instruct'][i]

    try:
        res = generator(data_ex, pad_token_id=generator.tokenizer.eos_token_id)
    except RuntimeError as e:
        res = e

    results.append(res)


with open('non-finetuned-non-selfinstruct-baseline-results.txt', 'w') as file:
    for item in results:
        file.write("%s\n" % item)

