import os
import re
import time
from typing import List

import numpy as np
import together
from dotenv import find_dotenv, load_dotenv

from conversation_clustering.plotting import plot_embeddings


def get_filepaths(directory):
    file_paths = []
    # Walk through all files and subdirectories in the given directory
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


def read_text_file(file_path):
    try:
        with open(file_path, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None


def generate_embeddings(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    """Generate embeddings from Together API.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """

    client = together.Together()
    outputs = client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return [x.embedding for x in outputs.data]


def sample_and_extract_embeddings(
    data_path: str, model_api_string: str, num_samples: int = -1, context_length=512
) -> np.ndarray:
    """Sample data examples and extract embeddings for each example.

    Args:
        data_path: str. A path to the data file. It should be in the .jsonl format.
        model_api_string: str. An API string for a specific embedding model of your choice.
        num_samples: int. The number of data examples to sample.
        context_length: int. The max context length of the model (model_api_string).

    Returns:
        embeddings: np.ndarray with num_samples by m where m is the embedding dimension.

    """
    max_num_chars = context_length * 4  # Assuming that each token is about 4 characters.
    embeddings = []
    count = 0
    print(f"Reading from {data_path}")
    with open(data_path, "r") as f:
        for line in f:
            try:
                # ex = json.loads(line)
                # read filepath
                total_chars = len(ex["text"])
                if total_chars < max_num_chars:
                    text_ls = [ex["text"]]
                else:
                    text_ls = [ex["text"][i : i + max_num_chars] for i in range(0, total_chars, max_num_chars)]
                embeddings.extend(
                    generate_embeddings(text_ls[: min(len(text_ls), num_samples - count)], model_api_string)
                )
                count += min(len(text_ls), num_samples - count)
            except Exception as e:
                print(f"Error occurred while loading the JSON file of {data_path} with the error message {e}.")
            if count >= num_samples:
                break
    return np.array(embeddings)


def main():
    # Define values.
    model = "togethercomputer/m2-bert-80M-8k-retrieval"  # You can use a different model.
    context_length = 8192
    num_samples = 50

    # get filepaths from directory
    filepaths = get_filepaths(DATA_DIR)

    list_text = []
    list_embedding = []
    list_text_name = []
    for f_path in filepaths:
        text = read_text_file(f_path)
        list_text.append(text)
        name = os.path.basename(f_path)
        name = re.sub(r"\d+_([^.]*)", r"\1", name)
        list_text_name.append(name)

        time.sleep(0.7)  # otherwise RateLimitError, because free version?
        embedding = generate_embeddings([text], model)
        embedding = np.array(embedding)
        embedding = embedding[:, np.newaxis]
        list_embedding.extend(embedding)

        # list_embedding.append(np.random.random((1, 768)))
    # x = np.concatenate(list_embedding)
    # embedding = generate_embeddings(list_text, model)
    save_path = "/Users/eric/Library/CloudStorage/Dropbox/git/github/conversation_clustering/data/plots/sample.png"
    plot_embeddings(list_embedding, n_components=2, names=list_text_name, perplexity=15, save_path=save_path)
    # read
    # embed_arr = sample_and_extract_embeddings(
    #     f_path,
    #     model_api_string=model,
    #     num_samples=num_samples,
    #     context_length=context_length
    # )
    # embedding_ls.append(embed_arr)
    # plot_embeddings(embedding_ls, n_components=2, names=filenames, perplextiy=15)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    together.api_key = os.environ.get("TOGETHER_API_KEY")
    # DATA_DIR = os.environ.get("DATA_DIR")
    DATA_DIR = "/Users/eric/Library/CloudStorage/Dropbox/git/github/conversation_clustering/data/raw/v3"
    main()
