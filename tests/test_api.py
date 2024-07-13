import os
import requests

def test_api_endpoints():
    base_url = os.getenv('API_BASE_URL', 'http://localhost:5000')

    endpoints = [
        '/',
        '/api/chat',
        '/api/llava',
        '/txt2img',
        '/list-models',
        '/install-model',
        '/uninstall-model',
        '/install'
    ]

    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        response = requests.get(url) if endpoint == '/' else requests.post(url)
        print(f"Testing {endpoint}: Status Code {response.status_code}")
        assert response.status_code in [200, 405], f"Failed: {endpoint}"

if __name__ == "__main__":
    test_api_endpoints()
