# Setup Script Documentation

## Overview
The `setup.sh` script is designed to automate the installation and configuration process for the Ollama API and Stable Diffusion. This script ensures that all prerequisites are met and that the environment is set up correctly to run the application.

## Prerequisites
Before running the `setup.sh` script, ensure that your system meets the following requirements:
1. **Operating System**: Ubuntu Linux or Macintosh (Windows is not supported).
2. **Hardware**:
   - 32 GB of RAM
   - 6 CPUs/vCPUs
   - 50 GB of Storage
   - NVIDIA GPU
3. **HuggingFace API Key**: A read-only HuggingFace API key is required. [Create one here](https://huggingface.co/docs/hub/security-tokens).

## Usage
To use the `setup.sh` script, follow these steps:

1. **Download the Script**:
   ```sh
   curl -OL https://raw.githubusercontent.com/tosin2013/ollama-api/main/setup.sh
   ```

2. **Make the Script Executable**:
   ```sh
   chmod +x setup.sh
   ```

3. **Run the Script**:
   ```sh
   ./setup.sh
   ```

## What the Script Does
The `setup.sh` script performs the following tasks:

1. **Update the System**:
   - Updates the package list and upgrades all installed packages to their latest versions.
   ```sh
   sudo apt-get update
   sudo apt-get upgrade -y
   ```

2. **Install Required Packages**:
   - Installs Python 3, pip, and other necessary dependencies.
   ```sh
   sudo apt-get install -y python3 python3-pip
   ```

3. **Install Python Dependencies**:
   - Navigates to the application directory and installs the required Python packages listed in `requirements.txt`.
   ```sh
   cd /path/to/app
   pip install -r requirements.txt
   ```

4. **Configure Firewall**:
   - Opens port 5000 to allow incoming traffic.
   ```sh
   sudo ufw allow 5000
   sudo ufw status
   ```

5. **Install CUDA Drivers** (if applicable):
   - Installs the latest CUDA drivers for GPU support.
   ```sh
   sudo apt-get install -y cuda-drivers
   ```

6. **Set Up HuggingFace API Key**:
   - Prompts the user to enter their HuggingFace API key and saves it to the `config.json` file.
   ```sh
   read -p "Enter your HuggingFace API key: " HF_API_KEY
   echo "{\"huggingface_api_key\": \"$HF_API_KEY\"}" > ./app/config.json
   ```

## Troubleshooting
If you encounter any issues while running the `setup.sh` script, consider the following:

1. **Permissions**: Ensure that the script is executable.
   ```sh
   chmod +x setup.sh
   ```

2. **Dependencies**: Verify that all required dependencies are installed.
   ```sh
   python3 --version
   pip --version
   ```

3. **Firewall**: Check if port 5000 is open.
   ```sh
   sudo ufw status
   ```

4. **API Key**: Ensure that the HuggingFace API key is correctly entered and saved in the `config.json` file.

## Support
For additional support, please refer to the main [README.md](../README.md) or contact the project maintainers.
