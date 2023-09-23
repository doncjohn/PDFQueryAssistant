from fastapi import FastAPI, HTTPException
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pickle

import os
os.environ["OPENAI_API_KEY"] = ""

# Initialize FastAPI app
app = FastAPI() 

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

# Endpoint for querying the stored embeddings
@app.post("/qna/")
async def query_embeddings(query: str):
    try:
        embeddings, docsearch = load_embeddings()

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        # Return the query result
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while querying the embeddings: " + str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
