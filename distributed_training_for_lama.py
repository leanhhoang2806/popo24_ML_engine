

# Run this code with
# `python distributed_training_for_lama.py`


import os
import time
import ray
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from numba import cuda

# import the Settings object here
# from llama_index.core import Settings
# to all the ray.remote() function
ray.init(ignore_reinit_error=True, logging_level="ERROR")

@ray.remote(num_gpus=1)
def load_documents(directory):
    print(f"Loading documents in process ID: {os.getpid()}")
    return SimpleDirectoryReader(directory).load_data()

@ray.remote(num_gpus=1)
def create_index(documents):
    from llama_index.core import VectorStoreIndex, Settings
    
    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # ollama
    Settings.llm = Ollama(model="mistral", request_timeout=360.0)
    try:
        index = VectorStoreIndex.from_documents(documents)
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

    return index

@ray.remote
def run_all(documents):
    from llama_index.core import Settings

    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # ollama
    Settings.llm = Ollama(model="mistral", request_timeout=360.0)

    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    response = query_engine.query("Is it ok to rob a bank ? Evaluate this text to only legal or illegal")
    print(response)

    # End timer
    end_time = time.time()

    execution_time = end_time - start_time
    execution_time_hours = execution_time / 3600
    print(f"Execution time: {execution_time_hours:.3f} hours")

# Start timer
start_time = time.time()

documents_future = load_documents.remote("data")

try:

    start_time = time.time()

    documents = SimpleDirectoryReader("data").load_data()
    run_all_future = run_all.remote(documents)
    ray.get(run_all_future)


finally:
    ray.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_hours = execution_time / 3600
    print(f"Execution time: {execution_time_hours:.3f} hours")
    cuda.select_device(0)
    cuda.close()
