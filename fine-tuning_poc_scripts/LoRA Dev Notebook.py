# Databricks notebook source
!pip install -q datasets accelerate loralib bitsandbytes

# COMMAND ----------

!pip install -q git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git

# COMMAND ----------

import os 
import torch
import datasets
import torch.nn as nn
import bitsandbytes as bnb

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

torch.cuda.is_available()

# COMMAND ----------


import requests 
url = "https://datasets-server.huggingface.co/first-rows?dataset=Open-Orca%2FOpenOrca&config=default&split=train"

response = requests.get(url)
data = response.json()

# COMMAND ----------

data

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(
    'bigscience/bloom-3b',
    torch_dtype = torch.float16,
    device_map = 'auto'
)

# COMMAND ----------

print(model)

# COMMAND ----------

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    target_modules=['query_key_value'],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# COMMAND ----------

datasets.utils.logging.disable_progress_bar()
sample_dataset = load_dataset('databricks/databricks-dolly-15k')

# COMMAND ----------



# COMMAND ----------


