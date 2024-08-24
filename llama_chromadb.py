from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
# from IPython.display import Markdown, display
import chromadb
import os

Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Make a chrome client
chromaClient = chromadb.EphemeralClient()
chroma_collection = chromaClient.create_collection("my_data")

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# documents = SimpleDirectoryReader("./data").load_data()

# Vector Store Setup
# vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)


CHROMA_DB_PATH = "./chroma_db"

if not os.path.exists(CHROMA_DB_PATH):
    print("NO CHROMA DB FOLDER FOUND, CREATING ONE")
    documents = SimpleDirectoryReader("./data").load_data()
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("my_data")
    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
else:
    print("FOUND A CHROMA DB FOLDER, NOT GOING TO REINDEX")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("my_data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine()
response = query_engine.query("What is the authors name?")
print(response)
# Store to disk
# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("my_data")
# vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store = vector_store)

# index = VectorStoreIndex.from_documents(docuemnts, context=storage_context, embed_model=embed_model)

# Load from disk

# query_engine = index.as_query_engine()
# response = query_engine.query("What is the authors name?")
# print(response)
