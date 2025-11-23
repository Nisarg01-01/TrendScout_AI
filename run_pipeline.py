import os
import subprocess
import argparse
import sys
from utils.neo4j_utils import get_neo4j_driver, clear_database
import config

# List of files to be deleted for a clean run
FILES_TO_DELETE = [
    config.ARTICLES_FILE,
    config.SNIPPETS_FILE,
    config.KPI_ENTITIES_FILE,
    config.ENTITY_MAP_FILE,
    os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")
]

# Sequence of scripts to execute for the pipeline
PIPELINE_STEPS = [
    "ingest_news.py",
    "ingest_jobs.py",
    "preprocess.py",
    "extract_llm.py",
    "dedupe_entities.py",
    "graph_build.py",
    "analysis_article_communities.py",
    "analysis_community_swot_summary.py",
    "rag_index.py"
]

def run_script(script_name):
    """Executes a python script and checks for errors."""
    print(f"\n{'='*20} Running: {script_name} {'='*20}")
    try:
        # Using sys.executable to ensure we use the same python interpreter
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("--- Stderr ---")
            print(result.stderr)
        print(f"{'-'*20} Finished: {script_name} {'-'*20}\n")
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERROR running {script_name} !!!!!!")
        print(e.stdout)
        print(e.stderr)
        raise

def clean_workspace():
    """Deletes generated data files and clears the database."""
    print("--- Starting Clean Up ---")
    
    # Delete data files
    for file_path in FILES_TO_DELETE:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"Skipped (not found): {file_path}")
            
    # Clear Neo4j Database
    try:
        driver = get_neo4j_driver()
        clear_database(driver)
        driver.close()
    except Exception as e:
        print(f"Could not clear Neo4j database: {e}")
        print("Please ensure Neo4j is running and credentials in config.py are correct.")

    print("--- Clean Up Complete ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TrendScout AI data pipeline.")
    parser.add_argument(
        '--clean',
        action='store_true',
        help="Perform a clean run by deleting all existing data and clearing the database."
    )
    args = parser.parse_args()

    if args.clean:
        clean_workspace()

    print("🚀🚀🚀 STARTING DATA PIPELINE 🚀🚀🚀")
    for step in PIPELINE_STEPS:
        run_script(step)
    print("✅✅✅ PIPELINE COMPLETED SUCCESSFULLY ✅✅✅")
