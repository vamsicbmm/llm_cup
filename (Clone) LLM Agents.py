# Databricks notebook source
# MAGIC %pip install -U langchain

# COMMAND ----------

# MAGIC %pip install -U openai

# COMMAND ----------

# MAGIC %pip install openai==0.28.1

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
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI
from langchain import PromptTemplate

from kfocr.assets.ipl.ocr_processor.LlpProcessor import LlpProcessor
from kfocr.assets.ipl.data_transform import DataTransform

# COMMAND ----------

# MAGIC %md
# MAGIC # Azure Open AI keys
# MAGIC The Azure keys are used to identify the Azure Open AI GPT model under the specified subscription.

# COMMAND ----------

openai.api_type = "azure" 
openai.api_base =  "https://kf-llm-ins-2.openai.azure.com/" # Your Azure OpenAI resource's endpoint value.
openai.api_key = "49b50a14e4e647c39d4522d8c0774119"
os.environ["OPENAI_API_KEY"] = "49b50a14e4e647c39d4522d8c0774119"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://kf-llm-ins-2.openai.azure.com/"

# COMMAND ----------

from langchain.tools import BaseTool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent

# COMMAND ----------

llm = AzureOpenAI(
        deployment_name='kf-gpt-35-turbo',
        model_name='gpt-35-turbo',
        temperature=0
)

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=1,
        return_messages=True
)

# COMMAND ----------

llm("What is an apple?")

# COMMAND ----------

engine_module_llp_data = [
  {
    "part_number": "335-009-306-0",
    "limit": 30000
  },
  {
    "part_number": "335-014-511-0",
    "limit": 30000
  },
  {
    "part_number": "335-006-414-0",
    "limit": 30000
  },
  {
    "part_number": "1275M37P02",
    "limit": 20000
  },
  {
    "part_number": "1589M66G02",
    "limit": 20000
  },
  {
    "part_number": "1590M59P01",
    "limit": 20000
  },
  {
    "part_number": "1588M89G03",
    "limit": 20000
  },
  {
    "part_number": "1319M25P02",
    "limit": 20000
  },
  {
    "part_number": "1385M90P04",
    "limit": 20000
  },
  {
    "part_number": "1282M72P05",
    "limit": 20000
  },
  {
    "part_number": "1475M29P02",
    "limit": 20000
  },
  {
    "part_number": "1864M91P02",
    "limit": 20000
  },
  {
    "part_number": "301-331-126-0",
    "limit": 25000
  },
  {
    "part_number": "301-331-225-0",
    "limit": 25000
  },
  {
    "part_number": "301-331-322-0",
    "limit": 25000
  },
  {
    "part_number": "301-331-427-0",
    "limit": 25000
  },
  {
    "part_number": "305-056-116-0",
    "limit": 25000
  },
  {
    "part_number": "301-330-066-0",
    "limit": 30000
  },
  {
    "part_number": "301-330-626-0",
    "limit": 25000
  },
  {
    "part_number": "335-009-306-0",
    "limit": 30000
  },
  {
    "part_number": "335-014-511-0",
    "limit": 24900
  },
  {
    "part_number": "335-006-414-0",
    "limit": 30000
  },
  {
    "part_number": "1275M37P02",
    "limit": 20000
  },
  {
    "part_number": "1589M66G02",
    "limit": 20000
  },
  {
    "part_number": "1590M59P01",
    "limit": 20000
  },
  {
    "part_number": "1588M89G03",
    "limit": 20000
  },
  {
    "part_number": "1319M25P02",
    "limit": 18000
  },
  {
    "part_number": "1385M90P04",
    "limit": 17300
  },
  {
    "part_number": "1282M72P05",
    "limit": 15800
  },
  {
    "part_number": "1475M29P02",
    "limit": 18500
  },
  {
    "part_number": "1864M91P02",
    "limit": 20000
  },
  {
    "part_number": "301-331-126-0",
    "limit": 25000
  },
  {
    "part_number": "301-331-225-0",
    "limit": 25000
  },
  {
    "part_number": "301-331-322-0",
    "limit": 25000
  },
  {
    "part_number": "301-331-427-0",
    "limit": 25000
  },
  {
    "part_number": "305-056-116-0",
    "limit": 25000
  },
  {
    "part_number": "301-330-066-0",
    "limit": 30000
  },
  {
    "part_number": "301-330-626-0",
    "limit": 25000
  }
]

parts_list = [_dict['part_number'] for _dict in engine_module_llp_data]
limit_list = [_dict['limit'] for _dict in engine_module_llp_data]

llp_processor = LlpProcessor()

# COMMAND ----------

# OCR output - first 5 page chunks summary 
# OCR processor - llp_tables output 

# COMMAND ----------

class OcrExtractionTool(BaseTool):
    name = "Ocr Extractor"
    description = "Use this tool to perform OCR (Optical Character Recognition) when you are asked to extract and summarise information in a document given via its URL. The input to this tool should only be the URL."
    # return_direct = True

    def _run(self, url: str):
        kf_ocr = KFocr(url)
        try:
            extracted_info = kf_ocr.get_page_text()
        except:
            return "The URL does not exist. "
        top_chunks = " ".join([element['text'] for element in extracted_info[:3]]) 
        return top_chunks
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class OcrLlpProcessorTool(BaseTool):
    name = "Llp Processor"
    description = "Use this tool to when you are asked to extract LLP details in a document given via its URL. The input to this tool should only be the URL."
    return_direct = True

    def _run(self, url: str):
        kf_ocr = KFocr(url)
        extracted_info = kf_ocr.analyze_dict
        llp_tables = llp_processor.process(extracted_info, parts_list, limit_list)
        return str(llp_tables)
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


tools = [OcrExtractionTool(), OcrLlpProcessorTool()]

# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

# agent = initialize_agent(
#     agent='zero-shot-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     early_stopping_method='generate', 
#     handle_parsing_errors = True
# )

# COMMAND ----------

print(agent.agent.llm_chain.prompt.messages[0].prompt.template)

# COMMAND ----------

sys_msg = """
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.  

In order to use the Ocr Extractor and Llp Processor tools, assistant will require the URL as an input provided by the user. In case no URL is provided in the question, assistant will not try to generate one on its own.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

agent.agent.llm_chain.prompt = new_prompt

# COMMAND ----------

In order to use the Ocr Extractor and Llp Processor tools, assistant will require the URL as an input provided by the user. In case no URL is provided in the question, assistant will not try to generate one on its own.

# COMMAND ----------

# agents calling another agents

# COMMAND ----------

sas_url = "https://virgodatalakestorage.blob.core.windows.net/tenant0001/raw/pdfs/1481642363_mini_pack%20(1).pdf?sv=2023-01-03&st=2023-11-01T08%3A09%3A52Z&se=2024-02-01T06%3A42%3A00Z&sr=b&sp=r&sig=5psFqELAkcP2kGrYK2i7fFkmhmkSCz9VRdQAqhO5TGU%3D"

agent(f"What does this document contain?")

# COMMAND ----------

import json 
a = [{"key1": 2, "key2": 3}]
b = json.dumps(a)
print(type(b))
c = json.loads(b)
print(type(c))

# COMMAND ----------

sas_url = "https://virgodatalakestorage.blob.core.windows.net/tenant0001/raw/pdfs/1481642363_mini_pack%20(1).pdf?sv=2023-01-03&st=2023-11-01T08%3A09%3A52Z&se=2024-02-01T06%3A42%3A00Z&sr=b&sp=r&sig=5psFqELAkcP2kGrYK2i7fFkmhmkSCz9VRdQAqhO5TGU%3D"
source = "721526"
kf_ocr = KFocr(sas_url)
extracted_info = kf_ocr.get_page_text()
print(extracted_info)

# COMMAND ----------

llp_tables

# COMMAND ----------

extracted_info['analyzeResult']['readResults'][0]

# COMMAND ----------


