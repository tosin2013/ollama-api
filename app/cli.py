import click
import requests

VALDI_ENDPOINT = 'http://localhost:5000'

@click.group()
def cli():
    pass

@cli.command()
@click.option('--model', required=True, help='Model name to install.')
def install_model(model):
    url = f"{VALDI_ENDPOINT}/install-model"
    response = requests.post(url, json={'model_name': model})
    click.echo(response.json())

@cli.command()
@click.option('--model', required=True, help='Model name to uninstall.')
def uninstall_model(model):
    url = f"{VALDI_ENDPOINT}/uninstall-model"
    response = requests.post(url, json={'model_name': model})
    click.echo(response.json())

@cli.command()
def list_models():
    url = f"{VALDI_ENDPOINT}/list-models"
    response = requests.get(url)
    click.echo(response.json())

@cli.command()
@click.option('--model', required=True, help='Model name to use.')
@click.option('--message', required=True, help='Message to send to the model.')
def chat(model, message):
    url = f"{VALDI_ENDPOINT}/api/chat"
    response = requests.post(url, json={'model': model, 'message': message})
    click.echo(response.json())

if __name__ == '__main__':
    cli()
