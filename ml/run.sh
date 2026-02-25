#!/usr/bin/env bash
cd "$(dirname "$0")/.." && PYTHONPATH=. uvicorn ml.main:app --host 0.0.0.0 --port 8000 --reload
