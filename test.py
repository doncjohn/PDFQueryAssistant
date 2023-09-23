from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pickle

import os
os.environ["OPENAI_API_KEY"] = ""

# Path to the local file for storing pre-computed embeddings
EMBEDDINGS_FILE = "embeddings.pkl"

# Load embeddings from a local file
def load_embeddings():
    try:
        with open(EMBEDDINGS_FILE, "rb") as embeddings_file:
            embeddings, docsearch = pickle.load(embeddings_file)
            return embeddings, docsearch
    except FileNotFoundError:
        return None, None

embeddings, docsearch = load_embeddings()

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "where is don working now"
docs = docsearch.similarity_search(query)
result = chain.run(input_documents=docs, question=query)
print(result)
