"""
DAG 03: Pipeline RAG / LLMOps - Atualização do Índice do Catálogo de Buracos Negros.

Extrai dados físicos dos modelos aprovados em Produção,
gera embeddings simulados e atualiza a base de busca semântica (RAG).
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

try:
    from airflow.sdk import dag, task, get_current_context
except ImportError:
    from airflow.decorators import dag, task
    from airflow.operators.python import get_current_context

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_index import EHTCatalogRAGIndex

try:
    import pendulum
    START_DATE = pendulum.datetime(2026, 7, 1, tz="UTC")
except Exception:
    START_DATE = datetime(2026, 7, 1)

BASE_DIR = Path("/tmp/eht_mlops_airflow_demo")


@dag(
    dag_id="03_eht_catalog_rag_refresh",
    start_date=START_DATE,
    schedule="@monthly",
    catchup=False,
    tags=["llmops", "rag", "catalog"],
    default_args={"retries": 1},
)
def eht_catalog_rag_refresh_dag():
    
    @task(task_id="extract_latest_production_model")
    def extract_latest_production_model() -> Dict[str, Any]:
        """
        Extracts the latest approved production model from the registry JSON db.
        """
        registry_db_path = BASE_DIR / 'model_registry.json'
        
        if not os.path.exists(registry_db_path):
            raise ValueError("No Model Registry database found. Run DAG 01 first.")
            
        with open(registry_db_path, 'r') as f:
            db = json.load(f)
            
        prod_run_id = db['active_production_model']
        if prod_run_id is None:
            raise ValueError("No active production model registered in Model Registry.")
            
        # Find run metadata
        prod_run = next((r for r in db['runs'] if r['run_id'] == prod_run_id), None)
        if prod_run is None:
            raise ValueError(f"Active production run {prod_run_id} metadata is missing.")
            
        print(f"Extracted latest production model: {prod_run_id}")
        return prod_run

    @task(task_id="update_vector_index")
    def update_vector_index(prod_run: Dict[str, Any]) -> str:
        """
        Generates text chunks and index embeddings, updating the RAG JSON database.
        """
        run_id = prod_run['run_id']
        parameters = prod_run['parameters']
        metrics = prod_run['metrics']
        fit_result = prod_run['fit_result']
        
        # Instantiate RAG Indexer
        rag = EHTCatalogRAGIndex(index_dir=str(BASE_DIR))
        
        # Generate text chunks & embeddings, save to index database
        num_chunks = rag.add_model_to_index(
            run_id=run_id,
            parameters=parameters,
            metrics=metrics,
            fit_result=fit_result
        )
        
        print(f"Updated index database with {num_chunks} chunks.")
        return str(rag.index_path)

    @task(task_id="validate_index_freshness")
    def validate_index_freshness(index_path: str, prod_run: Dict[str, Any]) -> str:
        """
        Validates that the index is fresh and correctly references the production run.
        """
        with open(index_path, 'r') as f:
            db = json.load(f)
            
        run_id = prod_run['run_id']
        
        # Check if any document contains the run_id
        contains_run = any(doc['metadata']['run_id'] == run_id for doc in db['documents'])
        
        if not contains_run:
            raise ValueError(f"Index freshness validation failed: Run {run_id} is missing from vector documents.")
            
        # Write validation metadata to check in the web UI
        status_report = {
            'fresh': True,
            'index_path': index_path,
            'document_count': len(db['documents']),
            'last_run_id': run_id,
            'validated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        web_rag_path = BASE_DIR / 'last_rag_web.json'
        with open(web_rag_path, 'w') as f:
            json.dump(status_report, f, indent=4)
            
        print(f"Index freshness validated: {len(db['documents'])} documents indexed. Freshness status: OK.")
        return index_path

    # Flow definitions
    prod_model = extract_latest_production_model()
    idx_path = update_vector_index(prod_model)
    validate_index_freshness(idx_path, prod_model)

# Instantiate DAG
eht_catalog_rag_refresh_dag()
