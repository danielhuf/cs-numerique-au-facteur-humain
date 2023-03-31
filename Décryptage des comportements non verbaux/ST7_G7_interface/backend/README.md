# Project ST7 Group 7 Backend

This is the repository for the frontend of Group 7's ST7 Project

## Setup

To run the project locally, you need to install the following:
- python 3.10

We recommend creating a virtual environment for the server.

To install the libraries, run

    pip install -r requirements.txt

You'll need to download the [AI model](**********):

    gdown **********

Then run the server with

    python app.py

The server will be listening in [http://localhost:5000](http://localhost:5000)

The file `deploy.sh` contains commands to run in a deployment environment, such as Azure.

## Deploy to Azure Cloud services

In case you need to deploy the app to a cloud server, here are some instructions to deploy. It didn't work for us, but we'll leave them here anyway.

You'll need an Azure account and the Azure CLI installed

Then run 

     az webapp up --runtime PYTHON:3.10 --sku B3 --logs --name <your-app-name>

Be careful, the specified type of virtual machine is B3 which costs money.

You'll need to specify to run `deploy.sh` in Startup Command, in the app settings.