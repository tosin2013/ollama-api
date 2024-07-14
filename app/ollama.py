import subprocess, shlex
import json

### This code is the automation and foundation for the support of multi model responses.  The code procures a question with content (chat history) and runs it through multiple models installed.
def listInstalledModels():
    curl_command = f'curl http://localhost:11434/api/tags'

    output = subprocess.check_output(curl_command, shell=True, encoding='utf-8')
    res = json.loads(output)

    # Extract only the 'name' attribute and remove ':latest'
    model_names = [model.get('name', '').replace(':latest', '') for model in res.get('models', [])]

    return model_names

def listModels():
    model_names = listInstalledModels()
    return {'model_names': model_names}

# Now you can print the result or do whatever you want with it
result = listModels()
print(result)


def run_model_generate(question, content):
    model_names = listInstalledModels()
    all_responses = {}

    for model in model_names:
        quoted_question = shlex.quote(question)
        quoted_content = shlex.quote(content)
        
        data_payload = {
            "model": model,
            "prompt": quoted_question,
            "content": quoted_content
        }

        json_data = json.dumps(data_payload)

        process = subprocess.Popen(['curl', 'http://localhost:11434/api/chat', '-d', json_data],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        
        output_str = output.decode('utf-8')

        print("Raw Output:", output_str)

        if process.returncode != 0:
            print(f"Error running command. Error message: {error.decode('utf-8')}")
            return

        try:
            responses = [json.loads(response)["response"] for response in output_str.strip().split('\n')]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response. Error message: {e}")
            return

        all_responses[model] = responses

    return all_responses

def run_model_chat(question, content):
    model_names = listInstalledModels()
    all_responses = {}

    for model in model_names:
        quoted_question = shlex.quote(question)
        quoted_content = shlex.quote(content)
        
        data_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": quoted_question},
                {"role": "assistant", "content": quoted_content}
            ],
            "stream": False
        }

        json_data = json.dumps(data_payload)

        process = subprocess.Popen(['curl', 'http://localhost:11434/api/chat', '-d', json_data],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        
        output_str = output.decode('utf-8')

        if process.returncode != 0:
            print(f"Error running command. Error message: {error.decode('utf-8')}")
            return

        try:
            response_json = json.loads(output_str)
            assistant_response = response_json.get('message', {}).get('content', '')
            all_responses[model] = assistant_response
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response. Error message: {e}")
            return

    return all_responses

results = run_model_chat("Question", "Content")

print(results)
