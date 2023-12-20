# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tuning Sentence Transformers embeddings
# MAGIC The task is to fine-tune embeddings for aviation specific domain.The following the steps to fine-tune the embeddings
# MAGIC 1. Take two openly available aviation engine mini-packs and one financial annual report of tesla motors.
# MAGIC 2. The two mini-pack text chucks will be labeled as positive score and the text chucks between mini-pack and tesla motors will be negatively scored.
# MAGIC 3. Using the above data, will be fine-tuning the embeddings and adding it to unity catalog.
# MAGIC 4. The purpose of chosing a car company is to differentiate the context of module in a car engine and a jet engine.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import essentials

# COMMAND ----------

import random
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC The data collected contains two pdfs containing engine mini-packs and one financial reports of tesla corp. The requirement is to define positively correlating sentences and negatively correlated sentences to fine-tune the embeddings. The intention of using automotive industy is to provide the context that module used in cars is different from modules used in aircrafts. The data will be prepared using following setup:
# MAGIC
# MAGIC 1. Perform OCR on the PDF files ingested using Structured streaming to extract the information.
# MAGIC 2. The sentences in mini-packs are going to be similar for positive samples.
# MAGIC 3. The sentences in financial reports and mini-packs are not similar for negative samples. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Global Variables

# COMMAND ----------

CATALOG_NAME = "DEV_CATALOG"
SCHEMA_NAME = "LLM_CUP"
PAGE_TEXT_TABLE_NAME = "PER_PAGE_TEXT"
MODEL_NAME = "aviation_tuned_embeddings"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the required tables
# MAGIC

# COMMAND ----------

 
page_text_table = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{PAGE_TEXT_TABLE_NAME}").toPandas() # converting to pandas for ease of processing
pdf_list = list(page_text_table["pdf_url"].unique())

# COMMAND ----------

pdf_list

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparing the data with positive and negative examples

# COMMAND ----------

positive_sets = [1,2] # set of pdfs with positive labels
negative_sets = [0,2] # set of pdfs with negative lables
positive_score_range = [0.7, 1] # value for positive label
negative_score_range = [0, 0.3] # value for negative label

# COMMAND ----------

# Filter pdf pages
filter_file_3 = page_text_table[page_text_table["pdf_url"] == pdf_list[negative_sets[0]]][page_text_table.page<20] # filter some samples from negative sample
filter_file_2 = page_text_table[page_text_table.page.isin(filter_file_3.page)][page_text_table["pdf_url"] == pdf_list[positive_sets[1]]] # filtering similar number of samples with same page numbers for positive samples
filter_file_1 = page_text_table[page_text_table.page.isin(filter_file_3.page)][page_text_table["pdf_url"] == pdf_list[positive_sets[0]]] # filtering similar number of samples with same page numbers for positive samples

# COMMAND ----------

# Preparing positive and negative datasets
loop_length = len(filter_file_1) # lopping through all the text
positive_dataset_list = []
negative_dataset_list = [] 
for i in range(loop_length):
    # selecting the first 100 characters for sentence creation
    text_1 = filter_file_1.iloc[i,:]["text"][:100] 
    text_2 = filter_file_2.iloc[i,:]["text"][:100]
    text_3 = filter_file_3.iloc[i,:]["text"][:100]
    # combining postive sentences and negative sentences
    pos_sentence = [text_1, text_2]
    neg_sentence = [text_1, text_3]
    # Adding label to the sentences
    pos_score = random.uniform(positive_score_range[0], positive_score_range[1])
    neg_score = random.uniform(negative_score_range[0], negative_score_range[1])
    pos_data = {"texts" : [text_1, text_2], "score":pos_score}
    neg_data = {"texts" : [text_1, text_3], "score":neg_score}
    positive_dataset_list.append(pos_data)
    negative_dataset_list.append(neg_data)

# COMMAND ----------

positive_dataset_list

# COMMAND ----------

negative_dataset_list

# COMMAND ----------

# combining positive_dataset and negative_dataset
data_set = positive_dataset_list + negative_dataset_list

# COMMAND ----------

# creating training samples using InputExample class
train_examples = []
for data in data_set:
    input_example = InputExample(texts = data["texts"], label = data["score"])
    train_examples.append(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tuning and logging the sentence transformer

# COMMAND ----------

#Define the model. Either from scratch of by loading a pre-trained model
  
model_registered_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
with mlflow.start_run():
    mlflow.transformers.autolog()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    #Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    #Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)
    mlflow.sentence_transformers.log_model(model, "model/fine-tuned-embeddings", registered_model_name= model_registered_name)
