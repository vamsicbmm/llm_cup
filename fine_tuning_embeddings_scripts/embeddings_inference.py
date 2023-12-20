# Databricks notebook source
# MAGIC %pip install langchain

# COMMAND ----------

import os
import requests
import json
import numpy as np
import pandas as pd
from langchain.vectorstores import FAISS, VectorStore
from langchain.embeddings import FakeEmbeddings
access_token = "dapi1a838640f6454fc76647e9d0193649b8-2"
def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-1035245746367987.7.azuredatabricks.net/serving-endpoints/aviation_tuned_embedding/invocations'
  headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
  ds_dict ={"inputs": dataset}
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# Define your custom embeddings class
class MyEmbeddings(FakeEmbeddings):
    
    def embed_query(self, text):

        embeddings = score_model([text])["predictions"][0]
        return embeddings
    
    def embed_documents(self, texts):
        embeddings = score_model(texts)["predictions"]
        return embeddings

# Create an instance of your custom embeddings class with your API url
my_embeddings = MyEmbeddings(size = 788)

# COMMAND ----------


