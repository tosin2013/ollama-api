## ollama-api-cli.py cli script

### API Endpoints
This directly interacts with the Backend Server hosted on VALDI.

- `/`: Serves the main chat interface.
- `/api/chat`: Handles chat messages sent to different language models.
- `/api/llava`: Specialized chat handler for the LLaVA model that includes image data.
- `/txt2img`: Endpoint for handling text-to-image generation requests.
- `/list-models`: Returns the list of available models installed on the server.
- `/install-model`: Installs a given model.
- `/uninstall-model`: Uninstalls a given model.
- `/install`: Endpoint used for initial setup, installing necessary components.

## CLI Usage
The `app/ollama-api-cli.py` script provides a command-line interface to interact with the Ollama API. Below are the available commands and their usage:

### Install Model
To install a model, use the following command:
```sh
python3 app/ollama-api-cli.py install-model --model <model_name>
```
Replace `<model_name>` with the name of the model you want to install.

### Uninstall Model
To uninstall a model, use the following command:
```sh
python3 app/ollama-api-cli.py uninstall-model --model <model_name>
```
Replace `<model_name>` with the name of the model you want to uninstall.

### List Models
To list all installed models, use the following command:
```sh
python3 app/ollama-api-cli.py list-models
```

### Chat with Model
To chat with a model, use the following command:
```sh
python3 app/ollama-api-cli.py chat --model <model_name> --message <message>
```
Replace `<model_name>` with the name of the model you want to chat with and `<message>` with the message you want to send.

## Credits ✨
This project would not be possible without continous contributions from the Open Source Community.
### Ollama
[Ollama Github](https://github.com/jmorganca/ollama)

[Ollama Website](https://ollama.ai/)

### @cantrell
[Cantrell Github](https://github.com/cantrell)

[Stable Diffusion API Server](https://github.com/cantrell/stable-diffusion-api-server)

### Valdi
Our preferred HPC partner  🖥️

[Valdi](https://valdi.ai/)

[Support us](https://valdi.ai/signup?ref=YZl7RDQZ)

### Replit
Our preferred IDE and deployment platform  🚀

[Replit](https://replit.com/)
