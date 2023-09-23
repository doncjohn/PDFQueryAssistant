from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import pickle

import os
os.environ["OPENAI_API_KEY"] = ""

EMBEDDINGS_FILE = "embeddings.pkl"

# location of the pdf file/files.
reader = PdfReader('CV.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

# Save embeddings to a local file
with open(EMBEDDINGS_FILE, "wb") as embeddings_file:
    pickle.dump((embeddings, docsearch), embeddings_file)