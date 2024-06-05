
# Working version for single training
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Start timer
start_time = time.time()

documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": "cuda"}, encode_kwargs={"device": "cuda", "batch_size": 100})

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=360.0)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("Is it ok to rob a bank ? Evaluate this text to only legal or illegal")
print(response)

# End timer
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
