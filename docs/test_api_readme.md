# Testing the API Endpoints

This document provides a guide on how to use the `tests/test_api.py` script to test the API endpoints of the application.

## Prerequisites

Before running the tests, ensure that the following prerequisites are met:

1. The application server is running and accessible at the specified URL.
2. The environment variable `API_BASE_URL` is set to the base URL of the application if it differs from the default (`http://localhost:5000`).

## Running the Tests

To run the tests, execute the following command in your terminal:

```sh
python tests/test_api.py
```

## What the Script Does

The `test_api.py` script performs the following actions:

1. Sets the base URL for the API requests.
2. Defines a list of endpoints to be tested.
3. Iterates over each endpoint, sending either a GET or POST request.
4. Prints the status code of each response.
5. Asserts that the status code is either 200 (OK) or 405 (Method Not Allowed), indicating that the endpoint is functioning as expected.

## Customizing the Base URL

If your application is hosted on a different URL, you can set the `API_BASE_URL` environment variable to override the default URL. For example:

```sh
export API_BASE_URL=http://your-custom-url:port
python tests/test_api.py
```

## Troubleshooting

If the tests fail, the script will print the status code of the failed request. Common issues include:

- The application server is not running.
- The endpoints are not correctly defined.
- The environment variable `API_BASE_URL` is not set correctly.

Ensure that the server is running and accessible, and that the endpoints are correctly defined in the script.
