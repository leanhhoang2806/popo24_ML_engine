
# To start this server. Running this code
# `serve run ray_cluster_model_deployment:legal_assistance_app`
# ======================================================

import time
import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import torch
import logging


# Define a request model for FastAPI
class QueryRequest(BaseModel):
    text: str

# Define a FastAPI app
app = FastAPI()

# Define the deployment class
@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class LegalAssistantModel:
    def __init__(self):
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
        # Load data and initialize the model
        self.documents = SimpleDirectoryReader("/home/hoang2/Documents/work/cheaper-ML-training/data/").load_data()
        # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
        # Settings.llm = Ollama(model="llama3", request_timeout=360.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")
        Settings.llm = Ollama(model="mistral", request_timeout=360.0)
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index.as_query_engine()

    @app.post("/query")
    async def query(self, request: QueryRequest):
        prompt = """
        You are a legal assistant to customer. 
        - Only answer the following context as legal or illegal
        - Only answer in one word. Don't extend anything
        - When in doubt, say None
        The following is the text use to evaluate: 
        """
        try:
            response = self.query_engine.query(prompt + request.text)
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

legal_assistance_app = LegalAssistantModel.bind()


# test code, the server did utilized GPU on deployment

# import requests

# # Define the URL of the endpoint
# url = "http://localhost:8000/query"

# # Define the payload
# payload = {
#     "text": "No, you cannot evict a tenant for refusing to pay rent. Evicting a tenant for this reason is not allowed"
# }

# # Define the headers
# headers = {
#     "Content-Type": "application/json"
# }

# # Send the POST request
# response = requests.post(url, json=payload, headers=headers)

# # Print the response
# response_json = response.json()
# print(response['response'])