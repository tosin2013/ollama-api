from flask import Flask, request, jsonify, render_template
import requests, random
import subprocess, sys
import json
import re
import base64
from PIL import Image
from io import BytesIO
import torch
import diffusers
import os

app = Flask(__name__)

VALDI_ENDPOINT = 'http://localhost:5000'
OLLAMA_INSTALL = False

# ------------> BACKEND FUNCTIONS <-------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/question', methods=['POST'])
def process_question():
    try:
        # Get the question from the request
        data = request.get_json()
        question = data.get('question', '')
        model = data.get('model', '')

        # Log the received question and model
        app.logger.info(f"Received question: {question} for model: {model}")

        # Run a command and capture the output
        result = run_model_question(question, model)
        app.logger.info(f"Model response: {result}")

        # Check for errors in the result
        if 'error' in result:
            app.logger.error(f"Error in model response: {result['error']}")
            return jsonify({"message": result['error']}), 400

        # Return the result as JSON
        return jsonify({"message": result})

    except ValueError as ve:
        app.logger.error(f"ValueError: {str(ve)}")
        return jsonify({"message": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"message": str(e)}), 500


@app.route('/api/vlm', methods=['POST'])
def vlm_model():
    data = request.get_json()
    model = data.get('model', '')
    prompt = data.get('prompt', '')
    image = data.get('image', '')

    result = run_vlm_question(model, prompt, image)
    return jsonify({"message": result})

@app.route('/api/pull', methods=['POST'])
def pull_model():
    data = request.get_json()
    model = data.get('model', '')

    result = run_pull_model(model)
    return jsonify({"message": result})

@app.route('/api/delete', methods=['DELETE'])
def delete_model():
    data = request.get_json()
    model = data.get('model', '')

    result = run_delete_model(model)
    return jsonify({"message": result})

@app.route('/api/install', methods=['GET'])
def install():
    global OLLAMA_INSTALL

    if not OLLAMA_INSTALL:
        response = install_ollama()
        OLLAMA_INSTALL = True
        return jsonify({'message': response})
    else:
        return jsonify({'message': 'OLLAMA_INSTALL is already set to True'})

@app.route('/api/list-models', methods=['GET'])
def listModels():
    res = listInstalledModels()
    return jsonify({'models': res})

######  HELPER FUNCTIONS   ######
def listInstalledModels():
    curl_command = f'curl http://localhost:11434/api/tags'
    output = subprocess.check_output(curl_command, shell=True, encoding='utf-8')
    res = json.loads(output)
    return res

def run_delete_model(model):
    curl_command = f'curl -X DELETE http://localhost:11434/api/delete -d \'{{"name": "{model}"}}\''
    output = subprocess.check_output(curl_command, shell=True, encoding='utf-8')
    response = json.loads(output)
    return response

def run_pull_model(model):
    curl_command = f'curl http://localhost:11434/api/pull -d \'{{"name": "{model}", "stream": false}}\''
    output = subprocess.check_output(curl_command, shell=True, encoding='utf-8')
    response = json.loads(output)
    return response

def run_vlm_question(model, prompt, image):
    curl_command = f'curl http://localhost:11434/api/generate -d \'{{"model": "{model}", "prompt": "{prompt}", "stream": false, "images": ["{image[0]}"]}}\''
    output = subprocess.check_output(curl_command, shell=True, encoding='utf-8')
    output_json = json.loads(output)
    responses = output_json.get("response", None)
    response_json = {'responses': responses}
    return response_json

def run_model_question(question, model):
    try:
        # Construct the curl command
        curl_command = f"curl -s -X POST http://localhost:11434/api/generate -H 'Content-Type: application/json' -d '{{\"model\": \"{model}\", \"prompt\": \"{question}\"}}'"

        app.logger.info(f"Executing curl command: {curl_command}")
        
        # Execute the curl command
        output = subprocess.check_output(curl_command, shell=True, encoding='utf-8')
        
        app.logger.info(f"Curl command output: {output}")

        # Check if the output contains a memory error
        if "model requires more system memory" in output:
            response_json = json.loads(output)
            error_message = response_json.get("error", "An error occurred")
            return {'responses': [], 'error': error_message}

        # Split the output into individual JSON objects
        output_lines = output.strip().split('\n')
        
        responses = []
        for line in output_lines:
            try:
                response_json = json.loads(line)
                responses.append(response_json.get("response", ""))
            except json.JSONDecodeError as e:
                app.logger.error(f"Error decoding JSON response: {e}")
                app.logger.error(f"Raw response: {line}")
                return {'responses': [], 'error': str(e)}
        
        concatenated_response = ' '.join(responses)
        return {'responses': [concatenated_response]}
    
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Error executing curl command: {e.output}")
        return {'responses': [], 'error': str(e)}
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return {'responses': [], 'error': str(e)}



def working_directory():
    current_directory = os.getcwd()
    file_to_find = "config.json"
    file_path = os.path.join(current_directory, file_to_find)
    return file_path

def install_ollama():
    try:
        curl_command = 'curl https://ollama.ai/install.sh | sh'
        subprocess.check_call(curl_command, shell=True, encoding='utf-8')
        return "Success"
    except subprocess.CalledProcessError as e:
        return str(e)

def retrieve_param(key, data, cast, default):
    return cast(data.get(key, default))

def pil_to_b64(input):
    buffer = BytesIO()
    input.save(buffer, 'PNG')
    output = base64.b64encode(buffer.getvalue()).decode('utf-8').replace('\n', '')
    buffer.close()
    return output

def b64_to_pil(input):
    return Image.open(BytesIO(base64.b64decode(input)))

def get_compute_platform(context):
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available() and context == 'engine':
        return 'mps'
    else:
        return 'cpu'

##################################################
# Engines

class Engine(object):
    def __init__(self):
        pass

    def process(self, kwargs):
        return []

class EngineStableDiffusion(Engine):
    def __init__(self, pipe, sibling=None, custom_model_path=None, requires_safety_checker=True):
        super().__init__()
        if sibling == None:
            self.engine = pipe.from_pretrained('runwayml/stable-diffusion-v1-5', use_auth_token=hf_token.strip())
        elif custom_model_path:
            if requires_safety_checker:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                safety_checker=sibling.engine.safety_checker,
                                                                                feature_extractor=sibling.engine.feature_extractor)
            else:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                feature_extractor=sibling.engine.feature_extractor)
        else:
            self.engine = pipe(
                vae=sibling.engine.vae,
                text_encoder=sibling.engine.text_encoder,
                tokenizer=sibling.engine.tokenizer,
                unet=sibling.engine.unet,
                scheduler=sibling.engine.scheduler,
                safety_checker=sibling.engine.safety_checker,
                feature_extractor=sibling.engine.feature_extractor
            )
        self.engine.to(get_compute_platform('engine'))

    def process(self, kwargs):
        output = self.engine(**kwargs)
        return {'image': output.images[0], 'nsfw': output.nsfw_content_detected[0]}

class EngineManager(object):
    def __init__(self):
        self.engines = {}

    def has_engine(self, name):
        return (name in self.engines)

    def add_engine(self, name, engine):
        if self.has_engine(name):
            return False
        self.engines[name] = engine
        return True

    def get_engine(self, name):
        if not self.has_engine(name):
            return None
        engine = self.engines[name]
        return engine

##################################################
# App Initialization

# Load and parse the config file:
try:
    config_file = open(working_directory(), 'r')
except:
    sys.exit('config.json not found.')

config = json.loads(config_file.read())
hf_token = config['hf_token']
if hf_token == None:
    sys.exit('No Hugging Face token found in config.json.')

custom_models = config['custom_models'] if 'custom_models' in config else []

# Initialize engine manager:
manager = EngineManager()
manager.add_engine('txt2img', EngineStableDiffusion(diffusers.StableDiffusionPipeline, sibling=None))
manager.add_engine('img2img', EngineStableDiffusion(diffusers.StableDiffusionImg2ImgPipeline, sibling=manager.get_engine('txt2img')))
manager.add_engine('masking', EngineStableDiffusion(diffusers.StableDiffusionInpaintPipeline, sibling=manager.get_engine('txt2img')))

for custom_model in custom_models:
    manager.add_engine(custom_model['url_path'],
                       EngineStableDiffusion(diffusers.StableDiffusionPipeline, sibling=manager.get_engine('txt2img'),
                                             custom_model_path=custom_model['model_path'],
                                             requires_safety_checker=custom_model['requires_safety_checker']))

# Define routes:
@app.route('/ping', methods=['GET'])
def stable_ping():
    return jsonify({'status': 'success'})

@app.route('/custom_models', methods=['GET'])
def stable_custom_models():
    if custom_models == None:
        return jsonify([])
    else:
        return custom_models

@app.route('/txt2img', methods=['POST', 'GET'])
def stable_txt2img():
    return _generate('txt2img')

@app.route('/img2img', methods=['POST'])
def stable_img2img():
    return _generate('img2img')

@app.route('/masking', methods=['POST'])
def stable_masking():
    return _generate('masking')

@app.route('/custom/<path:model>', methods=['POST'])
def stable_custom(model):
    return _generate('txt2img', model)

def _generate(task, engine=None):
    if engine == None:
        engine = task

    engine = manager.get_engine(engine)
    output_data = {}

    try:
        seed = retrieve_param('seed', request.form, int, 0)
        count = retrieve_param('num_outputs', request.form, int, 1)
        total_results = []

        for i in range(count):
            if seed == 0:
                generator = torch.Generator(device=get_compute_platform('generator'))
            else:
                generator = torch.Generator(device=get_compute_platform('generator')).manual_seed(seed)
            new_seed = generator.seed()
            prompt = request.get_json(force=True).get('prompt')
            args_dict = {
                'prompt': [prompt],
                'num_inference_steps': retrieve_param('num_inference_steps', request.form, int, 100),
                'guidance_scale': retrieve_param('guidance_scale', request.form, float, 7.5),
                'eta': retrieve_param('eta', request.form, float, 0.0),
                'generator': generator
            }
            if task == 'txt2img':
                args_dict['width'] = retrieve_param('width', request.form, int, 512)
                args_dict['height'] = retrieve_param('height', request.form, int, 512)
            if task == 'img2img' or task == 'masking':
                init_img_b64 = request.form['init_image']
                init_img_b64 = re.sub('^data:image/png;base64,', '', init_img_b64)
                init_img_pil = b64_to_pil(init_img_b64)
                args_dict['init_image'] = init_img_pil
                args_dict['strength'] = retrieve_param('strength', request.form, float, 0.7)
            if task == 'masking':
                mask_img_b64 = request.form['mask_image']
                mask_img_b64 = re.sub('^data:image/png;base64,', '', mask_img_b64)
                mask_img_pil = b64_to_pil(mask_img_b64)
                args_dict['mask_image'] = mask_img_pil

            pipeline_output = engine.process(args_dict)
            pipeline_output['seed'] = new_seed
            total_results.append(pipeline_output)

        output_data['status'] = 'success'
        images = [{'base64': pil_to_b64(result['image'].convert('RGB')), 'seed': result['seed'], 'mime_type': 'image/png', 'nsfw': result['nsfw']} for result in total_results]
        output_data['images'] = images

    except RuntimeError as e:
        output_data['status'] = 'failure'
        output_data['message'] = 'A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs for more details.'
        app.logger.error(str(e))
    
    return jsonify(output_data)

# ------------> FRONTEND INTERACTIONS <------------
@app.route('/api/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()

    if 'model' not in data or 'message' not in data:
        return jsonify({"error": "Missing 'model' or 'message' in JSON request"}), 400

    model = data['model']
    message = data['message']

    response = process_model_request(model=model, message=message)

    if 'error' in response:
        return jsonify(response), response[1]

    return jsonify(response), 200

@app.route('/api/llava', methods=['POST'])
def llavaChat():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()

    if 'model' not in data or 'message' not in data or 'image' not in data:
        return jsonify({"error": "Missing 'model', 'message', or 'image' in JSON request"}), 400

    model = data['model']
    message = data['message']
    image = data['image']

    try:
        response = requests.post(
            url=f"{VALDI_ENDPOINT}/api/vlm",
            headers={'Content-Type': 'application/json'},
            data=json.dumps({'prompt': message, 'model': model, 'image': [image]})
        )
        response.raise_for_status()
        return jsonify(response.json()), 200
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

def process_model_request(model, message):
    def post_to_valdi(model, message):
        try:
            response = requests.post(
                url=f"{VALDI_ENDPOINT}/api/question",
                headers={'Content-Type': 'application/json'},
                data=json.dumps({'question': message, 'model': model})
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}, 500

    if model == 'llama2':
        return post_to_valdi('llama2', message)
    elif model == 'mistral':
        return post_to_valdi('mistral', message)
    elif model == 'vlm':
        return post_to_valdi('vlm', message)
    else:
        try:
            response = requests.post(
                url=f"{VALDI_ENDPOINT}/api/question",
                headers={'Content-Type': 'application/json'},
                data=json.dumps({'question': message, 'model': model})
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Model '{model}' is unsupported, {e}"}, 404

@app.route('/txt2img-trigger', methods=['POST'])
def txt2img_route():
    try:
        request_data = request.get_json()

        prompt = request_data.get('prompt', '')
        seed = random.randint(1, 1000000)
        outputs = request_data.get('num_outputs', 1)
        width = request_data.get('width', 512)
        height = request_data.get('height', 512)
        steps = request_data.get('num_inference_steps', 10)
        guidance_scale = request_data.get('guidance_scale', 0.5)

        url = VALDI_ENDPOINT + '/txt2img'

        request_body = {
            "prompt": prompt,
            "seed": seed,
            "num_outputs": outputs,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale
        }

        headers = {
            'Content-Type': 'application/json',
        }

        response = requests.post(url, headers=headers, data=json.dumps(request_body))
        response_data = response.json()

        if response_data['status'] == 'success':
            image_data = response_data['images'][0]
            image_bytes = BytesIO(base64.b64decode(image_data['base64']))
            img = Image.open(image_bytes)
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_data = img_bytes.getvalue()

            return jsonify({
                'status': 'success',
                'message': 'Image data retrieved successfully',
                'image_data': base64.b64encode(img_data).decode('utf-8')
            })
        else:
            return jsonify({'error': f"Request failed: {response_data['message']}"}), 500
    except Exception as error:
        return jsonify({'error': f"Error: {error}"}), 500

@app.route('/list-models', methods=['GET'])
def list_models_route():
    try:
        url = VALDI_ENDPOINT + '/api/list-models'
        response = requests.get(url)
        response_data = response.json()

        return jsonify({
            'status': 'success',
            'message': 'Models retrieved successfully',
            'models': response_data['models']
        })
    except Exception as error:
        return jsonify({'error': f"Error: {error}"}), 500

@app.route('/install-model', methods=['POST'])
def install_model():
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        install_url = f"{VALDI_ENDPOINT}/api/pull"
        response = requests.post(install_url, json={'model': model_name})
        result = response.json().get('message')
        return jsonify({'message': result})
    except Exception as error:
        return jsonify({'error': str(error)}), 500

@app.route('/uninstall-model', methods=['POST'])
def uninstall_model():
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        uninstall_url = f"{VALDI_ENDPOINT}/api/delete"
        response = requests.delete(uninstall_url, json={'model': model_name})
        result = response.json().get('message')
        if result == None:
            result = "Model successfully uninstalled"
        return jsonify({"message": result})
    except Exception as error:
        return jsonify({'error': str(error)}), 500

@app.route('/install', methods=['GET'])
def install_get():
    try:  
        install_url = f"{VALDI_ENDPOINT}/api/install"
        response = requests.get(install_url)
        result = response.json().get('message')
        return jsonify({'message': result})
    except Exception as error:
        return jsonify({'error': str(error)}), 500

@app.route('/api/chat', methods=['GET'])
def get_chat():
    return jsonify({"message": "GET method is not supported for /api/chat"}), 405

def run_api():
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    run_api()
