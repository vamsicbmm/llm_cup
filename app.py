import os
import re
import json
import time
import openai
import pandas as pd
import textwrap
from typing import List, Union, Optional, Type, Dict, Tuple
from kfocr.core.kfocr_api import KFocr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain import PromptTemplate
from langchain.schema.document import Document
from langchain.agents import AgentType
from pydantic import BaseModel, Field

from azure.storage.blob import ContainerClient,BlobServiceClient,BlobServiceClient,BlobSasPermissions,generate_blob_sas,ContentSettings
from datetime import datetime, timedelta
from hashlib import sha256

from kfocr.assets.ipl.ocr_processor.LlpProcessor import LlpProcessor
from kfocr.assets.ipl.ocr_processor.EAProcessor import EAProcessor
from modules.shop_visit_forecast.model.foreacsting_models import ForecastingModels
from kfocr.assets.ipl.data_transform import DataTransform
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import BaseTool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

# Defining AzureOpenAI Keys
openai.api_type = "azure" 
openai.api_base =  "https://kf-llm-ins-2.openai.azure.com/" # Your Azure OpenAI resource's endpoint value.
openai.api_key = "49b50a14e4e647c39d4522d8c0774119"
os.environ["OPENAI_API_KEY"] = "49b50a14e4e647c39d4522d8c0774119"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://kf-llm-ins-2.openai.azure.com/"
os.environ["SERPAPI_API_KEY"] = "4ea6865054c8eb8608723170dd33e12314d8e182c10c16ee15565ccf03526e8e"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__c59a63b6ae494b5f97af20b35c96f711"
os.environ["LANGCHAIN_PROJECT"] = "tracing_agent"


STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=kfhackathonstorage;AccountKey=n5zFDF3EIX+q0J6x4xDlBjTvGmphRIHGnMh6Ed8Eo25oX1VYih0j5313cpYAUIVr+Nmda6Q8raCk+AStstbxZg==;EndpointSuffix=core.windows.net;"
STORAGE_CONTAINER_NAME    = "llmcup"

# Instantiate LLMs
llm = AzureChatOpenAI(
        deployment_name='kf-gpt-35-turbo-0613', model_name='gpt-35-turbo', temperature=0
)

llm_instruct = AzureOpenAI(
    deployment_name="kf-gpt-turbo-instruct", model_name="gpt-35-turbo-instruct", temperature= 0
)


@st.cache_resource(ttl="1h")
def configure_retriever(extracted_text, source):
    meta_data = {}
    docs = []
    for text_data in extracted_text:
        meta_data = {}
        page_data = text_data["text"]
        meta_data["page"] = text_data['page']
        meta_data["source"] = source
        document = Document(page_content= page_data, metadata= meta_data)
        docs.append(document)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 32)
    splits = text_splitter.split_documents(docs)
    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever( search_kwargs={"k": 3}, search_type="mmr")
    qa_template = """You are an aviation expert. Your task is to answer the user's question based on the provided information." If you don't understand the information, say you don't now. Don't generate any other answers. If you understand the information, answer the question in a polite and helpful manner. 
    Context: {context}
    Question: {question}
    Helpful Answer: 
    Your answer must be within the scope of the information provided. 
    """
    prompt = PromptTemplate(template=qa_template,
                                input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(llm= llm_instruct, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True,
                                    chain_type_kwargs = {"prompt": prompt})
    
    return qa_chain

@st.cache_resource(ttl="1h")
def ocr_extraction(document_url):
    kf_ocr = KFocr(document_url)
    extracted_info = kf_ocr.analyze_dict
    extracted_text = kf_ocr.get_page_text()
    return extracted_info, extracted_text

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    
    llm_response['result'] = wrap_text_preserve_newlines(llm_response['result'])
    return llm_response

# Upload files to Azure Blob Storage
def upload_file_to_blob(connection_string: str, container_name: str, blob_path: str, blob_content, content_type: str = "application/pdf"):
    # Instantiate a BlobServiceClient using a connection string
    blob_service_client = ContainerClient.from_connection_string(conn_str=connection_string,container_name = container_name)
    # Upload a blob to the container
    content_settings = ContentSettings(content_type=content_type)
    blob_service_client.upload_blob(blob_path, blob_content, overwrite=True, content_settings=content_settings, encoding='utf-8')
    return blob_service_client

def get_blob_sas_url(connection_string: str, container_name: str, blob_path: str, storedPolicyName:str=None, expiryInMinutes:int = 30):
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
    print("The sas url: ", sas_url)
    return sas_url


# Check if it's the first interaction
if "initialized" not in st.session_state:
    st.session_state.initialized = True
st.chat_message("assistant").write("👋 Welcome to the chatbot! Please upload the document you're interested in analyzing.")


uploaded_file = st.file_uploader("Choose the PDF file", accept_multiple_files=False)
if uploaded_file:
    bytes_data = uploaded_file.read()
    upload_file_to_blob(STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME, "app_data/"+uploaded_file.name, bytes_data)
    st.session_state.document_url = get_blob_sas_url(STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME, "app_data/"+uploaded_file.name)
else:
    st.info("Please upload the PDF to continue..")
    st.stop()
# st.session_state.document_url = st.text_input("Enter the URL of the document: ")

st.session_state.esn = st.text_input("Enter the Engine Serial Number: ")

# if not st.session_state["document_url"]:
#     st.info("Please upload PDF URl to continue..")
#     st.stop()

if "extracted_info" not in st.session_state and "retriever_chain" not in st.session_state:
    st.session_state.extracted_info, st.session_state.extracted_text = ocr_extraction(st.session_state.document_url)
    st.session_state.retriever_chain = configure_retriever(st.session_state.extracted_text, st.session_state.esn)


class KnowledgeBaseQuery(BaseModel):
    """Input Query to the Knowledge_Base_Tool function"""
    query: str = Field(..., description="The user's input question")

class ShopVisitForecastInput(BaseModel):
    """Inputs for the  Shop_Visit_Forecaster function."""
    llp_tables: List = Field(..., description = "The Llp details returned by the Llp_Processor function. You must first call the Llp_Processor function to get the llp_tables") 

class ShopVisitCostInput(BaseModel):
    """ Inputs for the Shop_Visit_Cost_Computer function"""
    shop_visit_forecast: Dict = Field(..., description="The Shop visit forecast values returned by the Shop_Visit_Forecaster function. You must first call the Shop_Visit_Forecast function to get the forecast values.")

class EAProcessorTool(BaseTool):
    name = "EA_Processor"
    description = "Use this tool when you are asked to extract the Engine details. This tool takes no input."
    # return_direct = True

    # args_schema: Type[BaseModel] = EAProcessorInput
    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    def _run(self):
        ea_processor = EAProcessor()
        ea_table = ea_processor.process(st.session_state.extracted_info)
        return ea_table
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    
class OcrLlpProcessorTool(BaseTool):
    name = "Llp_Processor"
    description = "Use this tool when you are asked to extract LLP details. You must also first call this tool when you are asked to forecast the next shop visit. This tool takes no input"


    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    def _run(self):
        llp_processor = LlpProcessor()
        engine_module_llp_data = pd.read_csv('https://kfhackathonstorage.blob.core.windows.net/llmcup/engine_module_llp_data%20(2).csv?sv=2021-10-04&st=2023-11-09T10%3A21%3A46Z&se=2023-12-10T10%3A21%3A00Z&sr=b&sp=r&sig=jwIIskStraH1XA25s3yZVjHNUg%2FqIkYWEN8hPE8F%2Bdo%3D')
        parts_list = engine_module_llp_data['part_number'].tolist()
        limit_list = engine_module_llp_data['limit'].tolist()
        llp_tables = llp_processor.process(st.session_state.extracted_info, parts_list, limit_list)
        llp_tables = llp_tables[0] if isinstance(llp_tables[0], list) else llp_tables
        return llp_tables
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    
class ShopVisitCostComputation(BaseTool):
    name = "Shop_Visit_Cost_Computer"
    description = "You must use this tool when you asked to compute costs for the next shop visits. You must first call the Shop_Visit_Forecaster function and use its output for this function. The input to this tool should be the entire dictionary returned by the Shop_Visit_Forecaster tool."


    
class ShopVisitForecastTool(BaseTool):
    name = "Shop_Visit_Forecaster"
    description = "Use this tool when you asked to forecast the next shop visit for an engine. You must first call the Llp_Processor function and use its output for this function. The input to this tool should be the entire full observation from the Llp Processor tool.The entire output of Llp Processor tool must be passed as an input to this tool"
    # return_direct = True

    args_schema: Type[BaseModel] = ShopVisitForecastInput

    def _run(self, llp_tables: List):
        forecasting_models = ForecastingModels()
        engine_module_llp_data = pd.read_csv('https://kfhackathonstorage.blob.core.windows.net/llmcup/engine_module_llp_data%20(2).csv?sv=2021-10-04&st=2023-11-09T10%3A21%3A46Z&se=2023-12-10T10%3A21%3A00Z&sr=b&sp=r&sig=jwIIskStraH1XA25s3yZVjHNUg%2FqIkYWEN8hPE8F%2Bdo%3D')
        engine_module_llp_data = engine_module_llp_data[engine_module_llp_data['enginefamily'] == 'V2500'][['part_number', 'limit']]
        parts_limits_list = engine_module_llp_data[['part_number', 'limit']].to_dict(orient='records')
    
        llp_table = llp_tables
        augmented_llp_table = []
        # engine_csn remains the same for each part number
        engine_csn = llp_table[0]['csn']
        for entry1, entry2 in zip(llp_table, parts_limits_list):
            llp_details_dict = {}
            if entry1['pn'] == entry2['part_number']:
                llp_details_dict['pn'] = entry1['pn']
                llp_details_dict['csn'] = entry1['csn']
                llp_details_dict['rem_cycles'] = int(entry2['limit']) - entry1['operated_cycles_1']
                augmented_llp_table.append(llp_details_dict)
        augmented_llp_df = pd.DataFrame(augmented_llp_table)
        shop_visit_forecast = forecasting_models.forecast_max_llp_usage_v1(augmented_llp_df, engine_csn)
        return shop_visit_forecast
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class KnowledgeBase(BaseTool):
    name = "Knowledge_Base_Tool"
    description = "Use this tool to answer any other generic questions about the underlying document. The input to this tool must be the full question asked by user."

    args_schema: Type[BaseModel] = KnowledgeBaseQuery

    def _run(self, query: str):
        llm_response = st.session_state.retriever_chain(query)
        final_response = process_llm_response(llm_response)
        return final_response
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

tools = [OcrLlpProcessorTool(), EAProcessorTool(), ShopVisitForecastTool(), KnowledgeBase()]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

st.chat_message("assistant").write("Thanks for sharing the document. Please put forward your question, and I'll do my best to help you.")
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)


