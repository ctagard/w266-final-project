import pandas as pd
import requests
from tqdm import tqdm


# System message
system = "You are an AI assistant with data science expertise ..."

# Load the data
# Define a function to get completion from your local LLaMA server
def get_llama_completion(prompt, example):
    input_text = f"Code Snippet: \n {example}\n Explanation: {prompt} \n Prompt:"
    data = {
        "model": "llama2",
        "prompt": input_text,
        "temperature": 0.4,
        "system": system,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=data)
        response_json = response.json()
        return response_json["response"]
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

# Load the data
data = pd.read_csv("processed_kernels_combined.csv")

# Iterate over the dataframe and get completions
enhanced_texts = []
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    enhanced_text = get_llama_completion(row['Prompt'], row['Example'])
    enhanced_texts.append(enhanced_text)

# Add the enhanced texts to the dataframe
data['Enhanced'] = enhanced_texts

# Save the updated dataframe
data.to_csv("processed_kernels_with_enhanced.csv", index=False)