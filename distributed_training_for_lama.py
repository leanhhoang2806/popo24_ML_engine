# This code work locally on it's own, now figure out how to run it on distributed system with GPU

import os
import time
import ray
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from numba import cuda

ray.init(ignore_reinit_error=True, logging_level="ERROR")

@ray.remote
def load_documents(directory):
    return SimpleDirectoryReader(directory).load_data()

@ray.remote
def create_index(documents):
    os.environ["OPENAI_API_KEY"] = "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"
    return VectorStoreIndex.from_documents(documents)

# Start timer
start_time = time.time()

documents_future = load_documents.remote("data")

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=360.0)

try: 
    documents = ray.get(documents_future)
    
    index_future = create_index.remote(documents)
    index = ray.get(index_future)

    query_engine = index.as_query_engine()
    response = query_engine.query("Is it ok to rob a bank? Evaluate this text to only legal or illegal")
    print(response)

finally:
    ray.shutdown()

    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_hours = execution_time / 3600
    print(f"Execution time: {execution_time_hours:.3f} hours")
    cuda.select_device(0)
    cuda.close()
