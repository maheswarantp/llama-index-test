from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--query",type=str, default="What is the authors name?")
args = parser.parse_args()

documents = SimpleDirectoryReader("./data").load_data()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# index = VectorStoreIndex.from_documents(documents)
# index.storage_context.persist()

# query_engine = index.as_query_engine()
# response = query_engine.query("What did author do growing up?")


PERSIST_DIR = './storage'
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query(args.query)
print(response)
