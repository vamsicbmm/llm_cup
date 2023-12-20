# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC Fine-tuning embeddings require text data which is present usually in unstrcutred data like PDFs. The fine-tuning is a continuos exercise where new set of data is regularly updated and the embeddings must be updated accordingly. Databricks Structured streaming is the best solution to achieve this. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Structured Streaming
# MAGIC The Autoloader is used to load the PDF files. The autoloader can be triggered whenever the new files are added. This way the data will be incrementally added and can be used to improve the embeddings. 

# COMMAND ----------

BASE_PATH = ""
