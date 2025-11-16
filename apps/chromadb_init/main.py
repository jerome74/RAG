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
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv

try:
    import requests
except Exception:
    requests = None

"""
env function

@param: name: str
"""

def env(name: str) -> Optional[str]:
    return os.environ.get(name)

"""
setup_logging function

@param: None
@return None
"""

def setup_logging() -> None:
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

"""
load_dataframe function

@param: csv_path: Path, encoding: str
@return DataFrame
"""


def load_dataframe(csv_path: Path, encoding: str) -> pd.DataFrame:
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
    return df

"""
generate_data_list function

@param: df: pd.DataFrame
@return list[list]
"""

def generate_data_list(df: pd.DataFrame) -> Dict[str, list]:
    
    try:
        docs = df["document"].tolist()
        ids = df["id"].tolist()
        metadatas = [
            {**json.loads(m), "title": t, "category": c, "source": s}
            for m, t, c, s in zip(df["metadata_json"], df["title"], df["category"], df["source"])
        ]
        return {"docs": docs, "ids": ids, "metadatas": metadatas}
    except Exception as e:
        logging.error(msg=f"error on retreive dataFrame information: {e}")
        raise e

"""
init_open_ai function

@param: df: None
@return embedding_functions.OpenAIEmbeddingFunction
"""

def init_open_ai() -> embedding_functions.OpenAIEmbeddingFunction:

    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=env("OPENAI_API_CYPHER"),
                    model_name=env("MODEL_NAME")
                )
        
        return openai_ef
    except Exception as e:
        logging.error(msg=f"error on init openai obj: {e}")
        raise e

"""
init_agent_mcp function

@param: df: None
@return MCPServerStreamableHttp
"""

def init_agent_mcp() -> MCPServerStreamableHttp:

    try:
        exa_search_mcp = MCPServerStreamableHttp(
            name="Exa Search MCP",
            params={
                "url": f"https://mcp.exa.ai/mcp?exaApiKey={env("EXA_KEY")}",
                "timeout": 30,
            },
            client_session_timeout_seconds=30,
            cache_tools_list=True,
            max_retry_attempts=1)
        
        return exa_search_mcp
    except Exception as e:
        logging.error(msg=f"error on init exa search obj: {e}")
        raise e

"""
connect_exa async function

@param: df: exa_search_mcp: MCPServerStreamableHttp
@return None
"""

async def connect_exa(exa_search_mcp: MCPServerStreamableHttp):
    
    try:
        await exa_search_mcp.connect()
    except Exception as e:
        logging.error(msg=f"error on connect exa search obj: {e}")
        raise e

"""
populate_collection function

@param: openai_ef: embedding_functions.OpenAIEmbeddingFunction, data: list[list]
@return None
"""

def populate_collection(openai_ef: embedding_functions.OpenAIEmbeddingFunction, data: Dict[str, list]) -> None:

    try:
        db_path=Path(env("DB_PATH"))
        db_path.mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(db_path)
        collection = chroma_client.get_or_create_collection(name=env("COLLECTION_NAME"))

        docs = data["docs"]
        ids = data["ids"]
        metadatas = data["metadatas"]

        vectors = openai_ef(docs)

        collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=vectors,)
        
        logging.info(f"count of {collection.count()} rows.")

        logging.info("test query on simple [Alarms], result equals 3")
        query = 'Alarms'
        query_embeddings = openai_ef([query])

        logging.info(f"result is:{collection.query(query_embeddings=query_embeddings,n_results=3)}")
    except Exception as e:
        logging.error(msg=f"error on populate chroma collection: {e}")
        raise e
    
"""
run function

@param: None
@return int
"""

def run() -> int:
    setup_logging()
    load_dotenv()

    try:
        df = load_dataframe(Path(env("CSV_PATH")), encoding=env("CSV_ENCODING"))
        data = generate_data_list(df)
        openai_ef = init_open_ai()
        populate_collection(data=data,openai_ef=openai_ef)

    except Exception as e:
        logging.error(f"Failed on run function: {e}")
        return 1

    return 0


if __name__ == "__main__":
    rc = run()
    sys.exit(rc)