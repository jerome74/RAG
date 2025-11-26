#!/usr/bin/env python3
"""
Initialize a ChromaDB persistent collection from a CSV, using OpenAI embeddings.

Configuration via environment variables:
  REQUIRED
    - OPENAI_API_KEY: OpenAI API key to create embeddings.

  OPTIONAL
    - COLLECTION_NAME: Target ChromaDB collection name (default: "derms_kb")
    - DB_PATH: Persistent directory for ChromaDB (default: "/data/vectordb")
    - MODEL_NAME: OpenAI embedding model (default: "text-embedding-3-small")
    - BATCH_SIZE: Embedding batch size (default: "128")
    - CSV_URL: If set, the CSV will be downloaded from this URL into /tmp/input.csv
    - CSV_PATH: Local path to the CSV (default: "/work/knowledgebase_for_chromadb.csv")
    - CSV_ENCODING: CSV file encoding (default: "ISO-8859-1")
    - CSV_DELIMITER: CSV delimiter (default: ",")
    - FAIL_ON_MISSING_COLUMNS: "true" to fail if expected columns are missing (default: "false")

Expected CSV columns (minimum):
  - "document": text content to embed
Optional columns if present:
  - "id": stable identifier per row; if missing, IDs will be generated
  - "metadata_json": JSON object per row with arbitrary metadata
  - "title", "category", "source": will be added to metadata if present
"""

import os
import sys
import json
import logging
import asyncio
import pandas as pd
import chromadb
import uvicorn
from chainlit.server import login

from chromadb.api.models import Collection
from chromadb.utils import embedding_functions
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, trace
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends


# Define request schema
class EstimateRequest(BaseModel):
    requirement: str

# Initialize FastAPI app
app = FastAPI()



# Rag Class
class Rag():
    openai_ef: embedding_functions.OpenAIEmbeddingFunction
    collection: Collection
    datas: Dict[str, list]
    df: pd.DataFrame

rag: Rag = Rag()

"""
##############################################################
env function

@param =  name: str
##############################################################
"""

def env(name: str) -> Optional[str]:
    return os.environ.get(name)

"""
##############################################################
setup_logging function

@param =  None
@return =  None
##############################################################
"""

def setup_logging() -> None:
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

"""
##############################################################
load_dataframe function

@param =  csv_path: Path, encoding: str
@return =  None
##############################################################
"""


def load_dataframe(csv_path: Path, encoding: str):
    logging.info(f"Current Working Directory (CWD): {Path.cwd()}")
    logging.info(f"Loading CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except Exception:
        logging.warning(f"Failed to decode with provided encoding Retrying with utf-8.")
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            logging.error(f"Failed load file csv with utf-8. {e}")
            raise e
        
    logging.info(f"Loaded {len(df)} rows.")

    rag.df = df

"""
##############################################################
generate_data_list function

@param =  None
@return =  None
##############################################################
"""

def generate_data_list():
    
    try:
        docs = rag.df["document"].tolist()
        ids = rag.df["id"].tolist()
        metadatas = [
            {**json.loads(m), "title": t, "category": c, "source": s}
            for m, t, c, s in zip(rag.df["metadata_json"], rag.df["title"], rag.df["category"], rag.df["source"])
        ]
        rag.datas = {"docs": docs, "ids": ids, "metadatas": metadatas}
    except Exception as e:
        logging.error(msg=f"error on retreive dataFrame information: {e}")
        raise e

"""
##############################################################
init_open_ai function

@param =  None
@return =  None
##############################################################
"""

def init_open_ai():

    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=env("OPENAI_API_KEY"),
                    model_name=env("MODEL_NAME")
                )
        
        rag.openai_ef = openai_ef
    except Exception as e:
        logging.error(msg=f"error on init openai obj: {e}")
        raise e

"""
##############################################################
populate_collection function

@param =  none
@return =  None
##############################################################
"""

def populate_collection():

    try:
        db_path=Path(env("DB_PATH"))
        db_path.mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(db_path)

        try:
            # Attempt to retrieve the collection
            collection = chroma_client.get_collection(name=env("COLLECTION_NAME"))
            print(f"Collection '{env("COLLECTION_NAME")}' exists.")
            # You can now work with the 'collection' object
            
        except Exception: 
            # Catch the exception raised when the collection is not found
            print(f"Collection '{env("COLLECTION_NAME")}' does not exist.")

            collection = chroma_client.get_or_create_collection(name=env("COLLECTION_NAME"))

            docs = rag.datas["docs"]
            ids = rag.datas["ids"]
            metadatas = rag.datas["metadatas"]

            vectors = rag.openai_ef(docs)

            collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=vectors,)
            
            logging.info(f"count of {collection.count()} rows.")

            logging.info("test query on simple [Alarms], result equals 3")
            query = 'Alarms'
            query_embeddings = rag.openai_ef([query])

            logging.info(f"result is:{collection.query(query_embeddings=query_embeddings,n_results=3)}")

        rag.collection = collection

    except Exception as e:
        logging.error(msg=f"error on populate chroma collection: {e}")
        raise e
    
"""
##############################################################
estimation_lookup_tool function_tool function

@param = query: str, max_results: int = 3
@return =  str
##############################################################
"""    

@function_tool
def estimation_lookup_tool(query: str, max_results: int = 3) -> str:

    query_embeddings = rag.openai_ef(["Audit Module,Grid Visualization and Alarms Module"])
    results = rag.collection.query(query_embeddings=query_embeddings, n_results=max_results)

    if not results["documents"][0]:
        return f"No requirement information found for: {query}"

    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            requirement = metadata["requirements"].title()
            team = metadata["team"].title()
            macro_activity = metadata["macro_activity"].title()
            activity = metadata["activity"].title()
            estimation_days = metadata["estimation_days"]
            
            formatted_results.append(
                f"The requirement: {requirement}, related macro activity: {macro_activity}, the team : {team}, estimate per activity: {activity}, estimates of the quantity of {estimation_days}"
            )

    
    logging.info("initialize Estimate Assistant...")
        
    return "Estimation Information:\n" + "\n".join(formatted_results)

"""
##############################################################
estimate_endpoint async function

@param =  estimate_agent: Agent, req: EstimateRequest
@return =  JSONResponse
##############################################################
"""    



@app.post("/estimate")
async def estimate_endpoint(req: EstimateRequest) -> JSONResponse:
    
          
          try: 
            logging.info("on estimate_endpoint")
            async with MCPServerStreamableHttp(
                name="Exa Search MCP",
                params={
                    "url": f"https://mcp.exa.ai/mcp?exaApiKey={env("EXA_API_KEY")}",
                    "timeout": 30,
                },
                client_session_timeout_seconds=30,
                cache_tools_list=True,
                max_retry_attempts=1) as exa_search_mcp:

                await exa_search_mcp.connect()

                estimate_agent = Agent(
                    name="Estimate Assistant",
                    instructions="""
                    You are a helpful estimate assistant giving out estimate information related requirements and team information.
                    You give concise answers.
                    * You follow this workflow:
                        0) First, use the estimation_lookup_tool to get the estimate information of the requirements. But only use the result if it's explicitly for the team, activity, requirements and estimantion days requested in the query.
                        1) If you couldn't find the exact match for the requirements or you need to look up the estimation, search the EXA web to figure out the exact information you nees.
                        Even if you have the estimation in the web search response, you should still use the estimation_lookup_tool to get the correct information of the estimation to make sure the information you provide is consistent.
                        2) Then, if necessary, use the estimation_lookup_tool to get the estimation information of the requirements.
                    * Even if you know the estimation of requirements, always use Exa Search to find the exact estimations.
                    * Once you know the estimation, use the estimation_lookup_tool to get the estimation information of the individual team the team has to be compose only use estimation_lookup_tool information.
                    * If the query is about requirement, in your final output give generate a table based on columns: Team_composition, persons per team, estimate and note. Fill whit the relative outcome values. Use a delimiter between any row
                    * Don't use the estimation_lookup_tool more than 10 times.
                    """,
                    tools=[estimation_lookup_tool],
                    mcp_servers=[exa_search_mcp],
                    )
                
                result = await Runner.run(
                            estimate_agent,
                            f"Based on the requirements: Audit Module,Grid Visualization and Alarms Module,how should a team be composed and how many day per activity :{req.requirement} ?"
                )
            
            lines = [line.strip() for line in result.final_output.split("\n") if line.strip()]

            return JSONResponse(content={"final_output": lines})
          except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)


"""
##############################################################
run function

@param =  None
@return =  int
##############################################################
"""

def run() -> int:
    setup_logging()
    load_dotenv()

    try:

        load_dataframe(Path(env("CSV_PATH")), encoding=env("CSV_ENCODING"))
            
        generate_data_list()
        init_open_ai()
        populate_collection()

        uvicorn.run(f"{env("MODULE_ASGI")}:app", host="0.0.0.0", port=8000, reload=True)


    except Exception as e:
        logging.error(f"Failed on run function: {e}")
        return 1

    return 0

# START POINT
if __name__ == "__main__":
    rc = run()
    sys.exit(rc)