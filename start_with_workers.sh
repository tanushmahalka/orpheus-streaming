#!/bin/bash

# Load environment variables
# Start uvicorn with multiple workers
echo "Starting FastAPI with multiple workers..."
uvicorn main_v2:app \
--host 0.0.0.0 \
--port 9090 \
--workers 12 \
--loop uvloop \
--http httptools \
--log-level warning \
--access-log 