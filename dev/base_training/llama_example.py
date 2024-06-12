
# Working version for single training
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Start timer
start_time = time.time()

documents = SimpleDirectoryReader("/home/hoang2/Documents/work/cheaper-ML-training/data/").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

prompt = """
You are a legal assistant to customer. 
- Only answer the following context as legal or illegal
- Only answer in one word. Don't extend anything
- When in doubt, say None
The following is the text use to evaluate: 
"""
response = query_engine.query(prompt + "No, you cannot evict a tennant for refusing to pay rent. Evicting a tenant for this reason is not allowed")
print(response)

# End timer
end_time = time.time()

execution_time = end_time - start_time
execution_time_hours = execution_time / 3600
print(f"Execution time: {execution_time_hours:.3f} hours")
