#!/usr/bin/env python3
"""
TrendScout AI environment "doctor".

Runs lightweight checks so users can quickly see what is missing:
- Python dependencies
- .env configuration
- Neo4j connectivity (optional)
- Ollama availability (optional)
- Presence of expected data artifacts
"""

import os
import socket
import sys

import config


def _print(status: str, msg: str):
    print(f"[{status}] {msg}")


def check_import(import_name: str, display: str | None = None) -> bool:
    display = display or import_name
    try:
        __import__(import_name)
        _print("OK", f"import {display}")
        return True
    except Exception as e:
        _print("FAIL", f"import {display}: {e}")
        return False


def check_env():
    py_ver = sys.version_info
    _print("INFO", f"Python: {sys.version.split()[0]}")
    if (py_ver.major, py_ver.minor) >= (3, 13):
        _print("WARN", "Python 3.13+ may not be supported by some ML dependencies (e.g., torch/sentence-transformers). Consider Python 3.11.")
    if os.path.exists(os.path.join(config.BASE_DIR, ".env")):
        _print("OK", ".env present")
    else:
        _print("WARN", ".env not found (copy .env.example -> .env)")

    if config.NEO4J_URI and config.NEO4J_USERNAME and config.NEO4J_PASSWORD:
        _print("OK", "Neo4j env vars set")
    else:
        _print("WARN", "Neo4j env vars missing (NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD)")


def check_neo4j():
    if not (config.NEO4J_URI and config.NEO4J_USERNAME and config.NEO4J_PASSWORD):
        return
    try:
        from neo4j import GraphDatabase
    except Exception:
        return

    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            session.run("RETURN 1").single()
        driver.close()
        _print("OK", "Neo4j connectivity")
    except Exception as e:
        _print("FAIL", f"Neo4j connectivity: {e}")


def check_ollama():
    if not check_import("ollama", "ollama"):
        return
    try:
        import ollama
        ollama.list()
        _print("OK", "Ollama reachable (ollama.list())")
    except Exception as e:
        _print("WARN", f"Ollama not reachable: {e}")


def check_data_files():
    expected = [
        config.ARTICLES_FILE,
        config.SNIPPETS_FILE,
        config.KPI_ENTITIES_FILE,
        os.path.join(config.DATA_DIR, "snippets_embeddings.parquet"),
    ]
    for path in expected:
        if os.path.exists(path):
            _print("OK", f"data file present: {os.path.relpath(path, config.BASE_DIR)}")
        else:
            _print("WARN", f"data file missing: {os.path.relpath(path, config.BASE_DIR)}")


def main():
    check_env()

    check_import("dateutil", "python-dateutil")
    check_import("pandas", "pandas")
    check_import("dotenv", "python-dotenv")
    check_import("requests", "requests")
    check_import("rapidfuzz", "rapidfuzz")
    check_import("neo4j", "neo4j")
    check_import("chromadb", "chromadb")

    check_neo4j()
    check_ollama()
    check_data_files()

    print("\nIf pandas fails with: 'dateutil: No module named dateutil'")
    print("- Run: python CODE/repair_dateutil.py")
    print("- Or reinstall: python -m pip install --force-reinstall --no-deps python-dateutil")


if __name__ == "__main__":
    main()
