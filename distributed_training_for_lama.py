
import os
import time
import ray
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

os.environ["OPENAI_API_KEY"] = "s"

service_context = ServiceContext.from_defaults(
    chunk_size=2048,
    llm=None,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
)

ray.init(ignore_reinit_error=True, logging_level="ERROR")

@ray.remote
def load_documents(directory):
    try:
        return SimpleDirectoryReader(directory).load_data(), None
    except Exception as e:
        return None, f"Error in load_documents: {e}"

@ray.remote
def create_index(documents):
    try:
        return VectorStoreIndex.from_documents(documents), None
    except Exception as e:
        return None, f"Error in create_index: {e}"

# Start timer
start_time = time.time()

documents_future = load_documents.remote("data")

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=360.0)

try: 
    documents, load_error = ray.get(documents_future)
    if load_error:
        raise RuntimeError(load_error)
    
    index_future = create_index.remote(documents)
    index, create_error = ray.get(index_future)
    if create_error:
        raise RuntimeError(create_error)

    query_engine = index.as_query_engine()
    response = query_engine.query("Is it ok to rob a bank? Evaluate this text to only legal or illegal")
    print(response)

finally:
    ray.shutdown()

# End timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
