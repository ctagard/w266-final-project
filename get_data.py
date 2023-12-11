import os
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi
import re
from bs4 import BeautifulSoup
import pandas as pd
import shutil
from black import format_str, FileMode

api = KaggleApi()
api.authenticate()
def remove_unwanted_content(text):
    """Removes HTML tags and Markdown image syntax."""
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text_no_html = re.sub(r"<.*?>", "", soup.get_text())

    # Remove Markdown image syntax
    text_no_images = re.sub(r"!\[.*?\]\(.*?\)", "", text_no_html)

    return text_no_images


def format_code_with_black(code_str):
    """Formats Python code using Black, ignoring lines that start with ! or %."""
    # Filter out lines starting with ! or %
    filtered_code = '\n'.join([line for line in code_str.split('\n') if not line.strip().startswith(('!', '%'))])

    try:
        formatted = format_str(filtered_code, mode=FileMode())
        return formatted
    except Exception as e:
        # If black fails to format, return the filtered code
        print(e)
        return filtered_code


def process_notebook(notebook_path):
    """Function to process a Jupyter Notebook and extract prompts and examples."""
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    prompt_examples = []
    current_prompt = []
    current_example = []

    for cell in notebook_content.cells:
        cell_type = cell.cell_type
        cell_content = cell.source

        if cell_type == "markdown":
            # Process and append Markdown content
            processed_content = remove_unwanted_content(cell_content)
            if current_example:
                prompt_examples.append((''.join(current_prompt), ''.join(current_example)))
                current_prompt = []
                current_example = []
            current_prompt.append(processed_content)
        elif cell_type == "code":
            # Process and append code content
            formatted_code = format_code_with_black(cell_content)
            current_example.extend(formatted_code)

    # After loop, save any remaining markdown-code pair
    if current_prompt and current_example:
        prompt_examples.append((''.join(current_prompt), ''.join(current_example)))

    return zip(*prompt_examples)  # Unzip into two lists


def download_and_process_kernel(kernel_ref, temp_dir="temp_kernel"):
    """Download a Kaggle kernel, process it, and then remove the temporary directory."""
    try:
        # Check if temp_dir exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # Removing directory if it exists

        # Make a temporary directory
        os.makedirs(temp_dir)

        # Download the kernel
        api.kernels_pull(kernel_ref, temp_dir, metadata=False)
        notebook_files = [f for f in os.listdir(temp_dir) if f.endswith('.ipynb')]

        if notebook_files:
            notebook_path = os.path.join(temp_dir, notebook_files[0])
            prompt, example = process_notebook(notebook_path)
            return prompt, example

    except Exception as e:
        print(f"Error processing kernel {kernel_ref}: {e}")
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # Deleting directory

    return None, None


notebook_index = 0  # Add a notebook counter

all_prompts = []
all_examples = []
all_notebook_indices = []
all_positions = []

for page in range(1, 100):

    # Get list of kernels to process
    kernel_list = api.kernels_list(search="ethics", page=page)

    for kernel in kernel_list:
        print(f"Processing kernel: {kernel.ref}")
        prompt, example = download_and_process_kernel(kernel.ref)

        notebook_index += 1  # Increment notebook index for each new kernel
        position = 0  # Reset position for each new notebook

        if prompt and example:
            for p, e in zip(prompt, example):
                position += 1  # Increment position for each prompt/example pair
                all_prompts.append(p)
                all_examples.append(e)
                all_notebook_indices.append(notebook_index)
                all_positions.append(position)

df = pd.DataFrame({
    'Notebook': all_notebook_indices,
    'Position': all_positions,
    'Prompt': all_prompts,
    'Example': all_examples
})

df.to_csv("processed_kernels_with_indices_ethics.csv", index=False)