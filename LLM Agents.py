# Databricks notebook source
# MAGIC %pip install -U langchain

# COMMAND ----------

# MAGIC %pip install -U openai

# COMMAND ----------

# MAGIC %pip install -U azure-storage-blob

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import os
import openai
from kfocr.core.kfocr_api import KFocr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.llms import AzureOpenAI
from langchain import PromptTemplate

# COMMAND ----------

# MAGIC %md
# MAGIC # Azure Open AI keys
# MAGIC The Azure keys are used to identify the Azure Open AI GPT model under the specified subscription.

# COMMAND ----------

openai.api_type = "azure" 
openai.api_base =  "https://kf-llm.openai.azure.com/" # Your Azure OpenAI resource's endpoint value.
openai.api_key = "d6e77295870f4e0fb5f44e2b96838801"
os.environ["OPENAI_API_KEY"] = "d6e77295870f4e0fb5f44e2b96838801"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
