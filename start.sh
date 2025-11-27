#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Stopping services..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT SIGTERM

# Activate virtual environment
source venv/bin/activate

# Start API in background
echo "Starting API..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to initialize..."
sleep 5

# Start Frontend
echo "Starting Frontend..."
streamlit run frontend/demo_app.py --server.port 8501

# Wait for processes
wait $API_PID
