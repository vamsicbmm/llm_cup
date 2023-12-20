# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC Fine-tuning embeddings require text data which is usaully found in unstructured data like PDFs. Embeddings must be updated based on the new incoming data so that the model is aware of the latest variations in the domain-specific context. DataBricks Structured Streaming is the way to go to achive the ingestion of the PDFs and update the latest incoming data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Structured Streaming
# MAGIC The Autoloader is used to perform the streaming on a blob storage where the PDF files are placed. It ingested the PDFs and pushes into ingest delta table

# COMMAND ----------

CATALOG_NAME = "dev_catalog"
SCHEMA_NAME = "llm_cup"
BASE_LOCATION = "abfss://tenant0001@virgodatalakestorage.dfs.core.windows.net/"
RAW_LOCATION = BASE_LOCATION + "llm_cup_files/"
CHECKPOINT_LOCATION = BASE_LOCATION+"llm_cup_cp/ingestion"

# COMMAND ----------

ingestion_stream = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "binaryFile") \
  .option("checkpointLocation", CHECKPOINT_LOCATION) \
  .option("cloudFiles.includeExistingFiles", "false") \
  .option("cloudFiles.validateOptions", "false") \
  .load(RAW_LOCATION)

# COMMAND ----------

(ingestion_stream.writeStream
   .format("delta")
   .outputMode("append")
   .option("checkpointLocation", "/llm_cup_write_cp")
   .trigger(once = True)
   .toTable(CATALOG_NAME+"."+SCHEMA_NAME+"."+"ingestion"))
