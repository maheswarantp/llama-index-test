from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel

import os


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm_llama3 = Ollama(model="llama3", request_timeout=60.0)

Settings.embed_model = embed_model
Settings.llm = llm_llama3

CONTEXT_QUERY_ENGINE_TEMPLATE = """Purpose: The primary role of this agent is to assist in reading resumes"""

PERSIST_DIR = "/workspace/llama/agent_dir"
if not os.path.exists(PERSIST_DIR):
    # index doesnt exist, create one for the documents
    print("Index not found, creating one...")
    documents = SimpleDirectoryReader("./data").load_data()
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    vector_index.storage_context.persist(PERSIST_DIR)
else:
    # index already exists, load that
    print(f"Index found, loading from directory: {PERSIST_DIR}")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    vector_index = load_index_from_storage(storage_context)

# Query Engine
query_engine = vector_index.as_query_engine(llm=llm_llama3)

# Writing tool
def write_tool(data):
    path = os.path.join("output", "output.txt")
    try:
        with open(path, "w") as f:
            f.write(data)
        
        return {"status": "wrote data to a file"}
    except Exception as e:
        return {"error": str(e)}

def lookup_table_for_job_openings(job_name):
    LOOKUP_TABLE = {
        "AI_ENGINEER": True,
        "SDE": True,
        "DEVOPS": False
    }
    try:
        return {"is_job_opening_available": LOOKUP_TABLE[job_name]}
    except Exception as e:
        return {"is_job_opening_available": LOOKUP_TABLE[job_name]}

tools = [
    QueryEngineTool(
        query_engine = query_engine,
        metadata = ToolMetadata(
            name="resume_documentation",
            description="this gives documentation for resume, contains all information regarding resumes"
        )
    ),
    FunctionTool.from_defaults(
        fn=write_tool,
        name="file_writer",
        description="This tool can write output information to a file and save it in the local filesystem"
    ),
    FunctionTool.from_defaults(
        fn=lookup_table_for_job_openings,
        name="job_opening_lookup_table",
        description="This tool will check if there is any job opening for any role which might be passed as a variable"
    )
]


agent = ReActAgent.from_tools(tools, llm=llm_llama3, verbose=True, context=CONTEXT_QUERY_ENGINE_TEMPLATE)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}: ", e)
    
    if retries >= 3:
        print("Unable to process request, try again...")
        continue
    

