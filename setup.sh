#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# set -e

# Function to display usage
usage() {
  echo "Usage: $0 [-i] [-c] [-a] [-s] [-p] [-h] YOUR_HUGGINGFACE_API_KEY"
  echo "  -i    Install dependencies"
  echo "  -c    Configure API key"
  echo "  -a    Allow port 5000"
  echo "  -s    Start Flask server"
  echo "  -p    Access API securely"
  echo "  -h    Display this help message"
  exit 1
}

# Parse command line options
while getopts "icasp" opt; do
  case ${opt} in
    i )
      INSTALL_DEPS=true
      ;;
    c )
      CONFIGURE_API=true
      ;;
    a )
      ALLOW_PORT=true
      ;;
    s )
      START_SERVER=true
      ;;
    p )
      ACCESS_API=true
      ;;
    h )
      usage
      ;;
    \? )
      usage
      ;;
  esac
done
shift $((OPTIND -1))

# Check if API key is provided as an argument
if [ -z "$1" ]; then
  echo "Error: API key is required."
  usage
fi

# Define variables
REPO_URL="https://github.com/tosin2013/ollama-api"
API_KEY="$1"
CONFIG_FILE="./app/config.json"
FLASK_APP="app.py"
LOG_FILE="app.log"
PORT=5000

# Function to install dependencies
install_dependencies() {
    echo "Checking and installing dependencies..."
    if ! command -v pip3 &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-pip curl jq
    else
        echo "pip3 is already installed. Skipping apt-get install."
    fi
    pip3 install -r app/requirements.txt
}

# Function to configure the API key
configure_api_key() {
    echo "Configuring API key..."
    cat <<EOF > $CONFIG_FILE
{
  "hf_token": "$API_KEY"
}
EOF
}

# Function to start the Flask server
start_flask_server() {
    echo "Checking for existing Flask server..."
    FLASK_PID=$(pgrep -f "python3 $FLASK_APP")
    if [ -n "$FLASK_PID" ]; then
        echo "Stopping existing Flask server with PID $FLASK_PID..."
        kill $FLASK_PID
    else
        echo "No existing Flask server found."
    fi

    echo "Starting Flask server..."
    cd app
    nohup python3 $FLASK_APP > $LOG_FILE 2>&1 &
    echo "Flask server started. Check $LOG_FILE for logs."
}

# Function to allow port 5000
allow_port() {
    echo "Allowing port $PORT..."
    echo "Configuring UFW..."
    yes | sudo ufw enable
    # add ssh to ufw
    sudo ufw allow ssh
    sudo ufw allow $PORT
    sudo ufw status
}

# Function to access the API securely
access_api() {
    echo "Accessing the API securely..."

    # Example: Sending a chat message
    curl -X POST http://localhost:$PORT/api/chat \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer $API_KEY" \
         -d '{
               "model": "llama2-uncensored",
               "prompt": "What is water made of?"
             }'

    # Example: Listing available models
    curl -X GET http://localhost:$PORT/list-models \
         -H "Authorization: Bearer $API_KEY"

    # Example: Generating an image from text
    curl -X POST http://localhost:$PORT/txt2img \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer $API_KEY" \
         -d '{
               "prompt": "A beautiful sunset over the mountains"
             }'
}

# Main script execution
main() {
    echo "Cloning the repository..."
    if [ ! -d "ollama-api" ]; then
        git clone $REPO_URL
        cd ollama-api
    else 
        echo "Directory already exists. Skipping cloning..."
        cd ollama-api
        git pull
    fi

    if [ "$INSTALL_DEPS" = true ]; then
        install_dependencies
    fi

    if [ "$CONFIGURE_API" = true ]; then
        configure_api_key
    fi

    if [ "$ALLOW_PORT" = true ]; then
        allow_port
    fi

    if [ "$START_SERVER" = true ]; then
        start_flask_server
    fi

    if [ "$ACCESS_API" = true ]; then
        echo "Waiting for the server to start..."
        sleep 5
        access_api
    fi
}

# Run the main function
main
