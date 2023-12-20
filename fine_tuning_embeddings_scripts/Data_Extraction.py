# Databricks notebook source
# MAGIC %md
# MAGIC # Data Extraction
# MAGIC The PDFs contains the text data required to fine-tune the embeddings. OCR is used to extract text from the PDFs. The extracted text is segregated per pages.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import essential libraries

# COMMAND ----------

from datetime import datetime, timedelta
from hashlib import sha256
from pyspark.sql.types import StringType
from azure.storage.blob import ContainerClient,BlobServiceClient,BlobServiceClient,BlobSasPermissions,generate_blob_sas,ContentSettings
import pandas as pd
from kfocr.core.kfocr_api import KFocr

# COMMAND ----------

# MAGIC %md
# MAGIC ## Global Variables

# COMMAND ----------

# Azure storage parameters
STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=virgodatalakestorage;AccountKey=NLSLW0uZAwBS1h05bYeCGHclG/XImJ4a2VlIgVbJnCdJOrlNVRK2gOUZGDPn8kGnHgUwglh/HdoR+AStUnDecQ==;EndpointSuffix=core.windows.net"
STORAGE_CONTAINER_NAME = "tenant0001"
BASE_PATH = "abfss://tenant0001@virgodatalakestorage.dfs.core.windows.net/"
CATALOG_NAME = "DEV_CATALOG"
SCHEMA_NAME = "LLM_CUP"
INGESTION_TABLE_NAME = "INGESTION"

# COMMAND ----------

@udf(returnType = StringType())
def get_relative_path(absolute_path, base_path=BASE_PATH):
    """
    Method to extract relative path from a absolute path
    Input Args:
    absolute_path(str): the absolute path
    base_path(str): the root path
    
    Returns:
    relative_path(str): relative path
    """
    relative_path = absolute_path.replace(base_path,"")
    return relative_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF Helper Functions

# COMMAND ----------

# Get file SAS URL from Azure Blob Storage
@udf(returnType = StringType())
def get_blob_sas_url(blob_path: str, connection_string: str = STORAGE_CONNECTION_STRING, container_name: str = STORAGE_CONTAINER_NAME, expiryInMinutes:int = 30):
        """
        Method to get the SAS url of file given the blob path
        Input Args:
        blob_path(str) : Path of the file
        connection_string(str): Connection string of the azure storage account.
        container_name(str) : Container name of the blob container
        expiryInMinutes(int): File SAS url active number of minutes 
        Returns:
        sas_url(str): SAS url of the file 
        """
        # Instantiate a BlobServiceClient using a connection string
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
        # Get a reference to a container
        container_client = blob_service_client.get_container_client(container_name)
        # Get a reference to a blob
        blob_client = container_client.get_blob_client(blob_path)
        # Genarate the blob permission
        blob_sas_permissions = BlobSasPermissions(read=True)      

        # Genrate blob sas
        sas_token = generate_blob_sas(
            blob_client.account_name,
            blob_client.container_name,
            blob_client.blob_name,
            account_key=blob_client.credential.account_key,
            permission=blob_sas_permissions,
            expiry=datetime.utcnow() + timedelta(minutes=expiryInMinutes)
        )
        sas_url = blob_client.url +'?'+ sas_token
       
        return sas_url

# COMMAND ----------

# MAGIC %md
# MAGIC ## OCR to extract text information from PDFs
# MAGIC Azure form recognizer is used to extract the text from the PDFs. To perform the OCR extraction it needs SAS urls of the files present in the blob storage. Relative path is needed to generate sas urls. The get_relative_path UDF will be used to get the relative path from the absolute path. Then the get_blob_sas_url UDF will be used to get the sas urls.
# MAGIC

# COMMAND ----------

# Reading the ingestion table
ingestion_table = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{INGESTION_TABLE_NAME}") 

# COMMAND ----------

# Get the relative path using UDF
ingestion_table = ingestion_table.withColumn("relative_path", get_relative_path(ingestion_table.path))

# COMMAND ----------

# drop the content column it is not required to generate the sas_url
ingestion_table = ingestion_table.drop("content")

# COMMAND ----------

# Generating SAS urls
sas_url_table = ingestion_table.withColumn("sas_url", get_blob_sas_url(ingestion_table.relative_path))

# COMMAND ----------

# collect the sas_urls into a list
sas_url_list = sas_url_table.select('sas_url').collect()
sas_url_list = [row.sas_url for row in sas_url_list] # using dot notation

# COMMAND ----------

# Looping through all the pdfs to extract text information per page
page_text_table = pd.DataFrame()
for sas_url in sas_url_list:
    kfocr = KFocr(sas_url) # Instantiating the OCR
    page_text = kfocr.get_page_text() # Extracting text per page 
    text_table = pd.DataFrame(page_text) # converting the extracted output to dataframe 
    text_table["pdf_url"] = sas_url # adding sas_url as identifier
    page_text_table = pd.concat((page_text_table, text_table)) # concat all the tables from all the pdfs

# COMMAND ----------

# Removing the appended index
page_text_table = page_text_table.reset_index().drop("index", axis=1)

# COMMAND ----------

page_text_table

# COMMAND ----------

#write page_text_table to a delta table
PAGE_TEXT_TABLE_NAME = "PER_PAGE_TEXT" 
page_text_df = spark.createDataFrame(page_text_table)
page_text_df.write.mode("append").saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{PAGE_TEXT_TABLE_NAME}")
