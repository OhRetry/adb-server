call conda activate adb-server
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --log-config logging.yaml