# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tuning Sentence Transformers embeddings
# MAGIC The task is to fine-tune embeddings for aviation specific domain.The following the steps to fine-tune the embeddings
# MAGIC 1. Take two openly available aviation engine mini-packs and one financial annual report of tesla motors.
# MAGIC 2. The two mini-pack text chucks will be labeled as Positively correlated and the text chucks between mini-pack and tesla motors will be negatively labelled.
# MAGIC 3. Using the above data, will be fine-tuning the embeddings.
# MAGIC 4. The purpose of chosing a car company is to differentiate the context of module in a car engine and a jet engine.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import essentials

# COMMAND ----------

import random
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from kfocr.core.kfocr_api import KFocr

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC The data collected contains two pdfs containing engine mini-packs and one financial reports of tesla corp. The data will be prepared using following setup:
# MAGIC 1. The sentences in mini-packs are going to be similar. 
# MAGIC 2. The sentences in financial reports and mini-packs are not similar. 

# COMMAND ----------

## SAS urls for OCR to read PDFs
mini_pack_1 = "https://virgodatalakestorage.blob.core.windows.net/tenant0001/raw/pdfs/1481642363_mini_pack%20(1).pdf?sv=2023-01-03&st=2023-11-01T08%3A09%3A52Z&se=2024-02-01T06%3A42%3A00Z&sr=b&sp=r&sig=5psFqELAkcP2kGrYK2i7fFkmhmkSCz9VRdQAqhO5TGU%3D"
mini_pack_2 = "https://virgodatalakestorage.blob.core.windows.net/tenant0001/raw/pdfs/1617667203_esn_31109_minipack_opt.pdf?sv=2023-01-03&st=2023-11-01T08%3A11%3A05Z&se=2024-02-02T08%3A11%3A00Z&sr=b&sp=r&sig=hTUjXvqFl6qqGFeWZ0%2Bg8MmzQ5n9IKDosubNNIwE%2BGc%3D"
finance_statement = "https://virgodatalakestorage.blob.core.windows.net/tenant0001/raw/pdfs/tsla-10k_20201231-gen-10-30.pdf?sv=2023-01-03&st=2023-11-01T08%3A11%3A29Z&se=2024-02-02T08%3A11%3A00Z&sr=b&sp=r&sig=NadcofzCNkoiPubt5cwEhIRZ0mtwdCpmnoZmqmZ6u4s%3D"

# COMMAND ----------

# combining the pdfs
pdf_list = [mini_pack_1, mini_pack_2, finance_statement]

# COMMAND ----------

# MAGIC %md
# MAGIC ## OCR
# MAGIC OCR to extract text from the PDF

# COMMAND ----------

# Looping through all the pdfs
page_text_table = pd.DataFrame()
for pdf in pdf_list:
    kfocr = KFocr(pdf) # Instantiating the OCR
    page_text = kfocr.get_page_text() # Extracting text per page 
    text_table = pd.DataFrame(page_text) # converting extracted the output to dataframe 
    text_table["pdf_url"] = pdf # adding sas_url as identifier
    page_text_table = pd.concat((page_text_table, text_table)) # concat al the tables from all the pdfs

# COMMAND ----------

page_text_table = page_text_table.reset_index().drop("index", axis=1)

# COMMAND ----------

page_text_table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparing the data with positive and negative examples

# COMMAND ----------

positive_sets = [0,1] # set of pdfs with positive labels
negative_sets = [0,2] # set of pdfs with negative lables
positive_label_range = [0.7, 1] # value for positive label
negative_label_range = [0, 0.3] # value for negative label

# COMMAND ----------

# Filter pdf pages
filter_file_3 = page_text_table[page_text_table["pdf_url"] == pdf_list[negative_sets[1]]].sample(n=20, random_state = 2).sort_index()
filter_file_2 = page_text_table[page_text_table.page.isin(filter_file_3.page)][page_text_table["pdf_url"] == pdf_list[positive_sets[1]]]
filter_file_1 = page_text_table[page_text_table.page.isin(filter_file_3.page)][page_text_table["pdf_url"] == pdf_list[positive_sets[0]]]

# COMMAND ----------

filter_file_1

# COMMAND ----------

loop_length = len(filter_file_1)
positive_dataset_list = []
negative_dataset_list = [] 
for i in range(loop_length):
    text_1 = filter_file_1.iloc[i,:]["text"][:100]
    text_2 = filter_file_2.iloc[i,:]["text"][:100]
    text_3 = filter_file_3.iloc[i,:]["text"][:100]
    pos_sentence = [text_1, text_2]
    neg_sentence = [text_1, text_3]
    pos_label = random.uniform(positive_label_range[0], positive_label_range[1])
    neg_label = random.uniform(negative_label_range[0], negative_label_range[1])
    pos_data = {"texts" : [text_1, text_2], "label":pos_label}
    neg_data = {"texts" : [text_1, text_3], "label":neg_label}
    positive_dataset_list.append(pos_data)
    negative_dataset_list.append(neg_data)

# COMMAND ----------

positive_dataset_list

# COMMAND ----------

negative_dataset_list

# COMMAND ----------

data_set = positive_dataset_list + negative_dataset_list

# COMMAND ----------


train_examples = []
for data in data_set:
    input_example = InputExample(texts = data["texts"], label = data["label"])
    train_examples.append(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tuning the transformer

# COMMAND ----------

#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)

# COMMAND ----------

v1 = model.encode("The engine module hpt")
v2 = model.encode("The car module gpt")

# COMMAND ----------


