import os

import together
from dotenv import find_dotenv, load_dotenv


def main():
    # Define values.
    model = "togethercomputer/m2-bert-80M-8k-retrieval"  # You can use a different model.
    context_length = 8192
    num_samples = 50


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    together.api_key = os.environ.get("TOGETHER_API_KEY")
    # DATA_DIR = os.environ.get("DATA_DIR")
    DATA_DIR = "/Users/eric/Library/CloudStorage/Dropbox/git/github/conversation_clustering/data/raw/v3"
    main()


# embedding_ls = []
# for f_name in filenames:
#     embed_arr = sample_and_extract_embeddings(
#     	os.path.join(data_dir, f_name),
# model_api_string=model,
# num_samples=num_samples,
# context_length=context_length
#     )
# embedding_ls.append(embed_arr)
# plot_embeddings(embedding_ls, n_components=2, names=filenames, perplextiy=15)


# Streamlit app
def main():
    st.title("Cluster Visualization App")

    # Sample data
    data_source1 = np.random.rand(10, 50)  # Replace with your actual data
    data_source2 = np.random.rand(15, 50)  # Replace with your actual data

    embed_arr_ls = [data_source1, data_source2]
    names = ["Data Source 1", "Data Source 2"]

    # Sidebar inputs
    n_components = st.sidebar.slider("Number of components", min_value=2, max_value=10, value=2)
    perplexity = st.sidebar.slider("Perplexity", min_value=5, max_value=50, value=30)

    # Plot clusters
    plot_embeddings(embed_arr_ls, n_components, names, perplexity)


if __name__ == "__main__":
    main()
