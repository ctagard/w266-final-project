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


def format_prompt(row):
    sys_prompt = f"""
    <s>[INST] <<SYS>>
    Here is some python code with some comments. Generate an instruction that would result in the code. 
    Examples
    ========
    code:
    ```
    import numpy as np
    ``` 
    instruction:
    Write Python code to import the NumPy library, and name it as np.

    code:
    ```
    df = pd.read_csv('filename.csv')
    ``` 
    instruction:
    Write Python code to read a CSV file named 'filename.csv' into a pandas DataFrame named df.

    code:
    ```
    import matplotlib.pyplot as plt
    df['column_name'].plot(kind='hist')
    plt.show()
    ```
    instruction:
    Write Python code to import the matplotlib.pyplot module, plot a histogram of the values in the column 'column_name' of a DataFrame df, and display the plot.
    <</SYS>>

    code:
    ```
    {row['code']}
    ```
    instruction:
    [/INST]
    """
test = pd.read_csv("train.csv")
test_dataset = Dataset.from_pandas(test)
ds = test_dataset.map(lambda sample: {'formatted_instruction': format_prompt(sample)})



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

for i in tqdm(range(len(test_dataset['formatted_instruction']))):
    data_ex = test_dataset['formatted_instruction'][i]

    try:
        res = generator(data_ex, pad_token_id=generator.tokenizer.eos_token_id)
    except RuntimeError as e:
        res = e

    results.append(res)


with open('backtranslation.txt', 'w') as file:
    for item in results:
        file.write("%s\n" % item)

