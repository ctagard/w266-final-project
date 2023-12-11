from datasets import load_dataset, Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import deepspeed
import os
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
from peft import AutoPeftModelForCausalLM, PeftModel

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))

turing_dataset = "w266finalproject/turing-60k-instruct"
dataset = load_dataset(turing_dataset, split='train')

def format_instruction(sample):
    return f"""<s>[INST] <<SYS>>\nHere is some python code with some comments. Generate an instruction that would result in the code. \nExamples\n========\ncode:\n```\nimport numpy as np\n``` \ninstruction:\nWrite Python code to import the NumPy library, and name it as np.\n\ncode:\n```\ndf = pd.read_csv('filename.csv')\n``` \ninstruction:\n Write Python code to read a CSV file named 'filename.csv' into a pandas DataFrame named df.\n\ncode:\n```\nimport matplotlib.pyplot as plt\ndf['column_name'].plot(kind='hist')\nplt.show()\n```\ninstruction:\nWrite Python code to import the matplotlib.pyplot module, plot a histogram of the values in the column 'column_name' of a DataFrame df, and display the plot.\n<</SYS>>\n\ncode:\n```\n{sample['Example']}```\ninstruction:\n [/INST]</s>"""


# Apply the formatting function to each sample in the dataset
dataset = dataset.map(lambda sample: {"formatted_instruction": format_instruction(sample)})


test = pd.read_csv("test_formatted.csv")
print(test.head())
test_dataset = Dataset.from_pandas(test)



model_id = "codellama/CodeLlama-7b-Instruct-hf"

def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
    model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    use_safetensors=True,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


model, tokenizer = create_model_and_tokenizer()
finetuned_model = PeftModel.from_pretrained(model, "../turing/turing-7b-2epoch-12-08-2023")

generator = pipeline('text-generation', 
                     model=finetuned_model,
                     tokenizer=tokenizer,
                     device_map="auto",
                     max_new_tokens=256)

#generator.model = deepspeed.init_inference(generator.model,
#
#                                           mp_size=world_size,
#                                           dtype=torch.float16,
#                                           max_out_tokens=2048,
#                                           replace_with_kernel_inject=True)
generator.tokenizer.padding_side='left'


results = []

for i in tqdm(range(len(test_dataset['baseline_non_instruct']))):
    data_ex = test_dataset['baseline_non_instruct'][i]

    try:
        res = generator(data_ex, pad_token_id=generator.tokenizer.eos_token_id)
    except RuntimeError as e:
        res = e

    results.append(res)


with open('fine-tuned-non-self-instruct-baseline.txt', 'w') as file:
    for item in results:
        file.write("%s\n" % item)

