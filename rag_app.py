import os
import openai
import re
import json
import time
import textwrap
import tempfile
import streamlit as st
from typing import List, Union, Optional, Type, Dict, Tuple
from kfocr.core.kfocr_api import KFocr
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.agents import AgentType
from pydantic import BaseModel, Field
from kfocr.assets.ipl.ocr_processor.LlpProcessor import LlpProcessor
from kfocr.assets.ipl.ocr_processor.EAProcessor import EAProcessor
from modules.shop_visit_forecast.model.foreacsting_models import ForecastingModels
from kfocr.assets.ipl.data_transform import DataTransform
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import BaseTool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

from azure.storage.blob import ContainerClient,BlobServiceClient,BlobServiceClient,BlobSasPermissions,generate_blob_sas,ContentSettings
from datetime import datetime, timedelta
from hashlib import sha256

st.set_page_config(page_title="FinTwin Agent", page_icon="🦜")
st.title("🦜 FinTwin Skybot - Chat with documents")

# Defining AzureOpenAI Keys
openai.api_type = "azure" 
openai.api_base =  "https://kf-llm-ins-2.openai.azure.com/" # Your Azure OpenAI resource's endpoint value.
openai.api_key = "49b50a14e4e647c39d4522d8c0774119"
os.environ["OPENAI_API_KEY"] = "49b50a14e4e647c39d4522d8c0774119"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://kf-llm-ins-2.openai.azure.com/"

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

    return retriever

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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# Check if it's the first interaction
if "initialized" not in st.session_state:
    st.session_state.initialized = True
st.chat_message("assistant").write("👋 Welcome to the chatbot! Please upload the document you're interested in analyzing.")


uploaded_file = st.sidebar.file_uploader(
    label="Upload the PDF file", type=["pdf"], accept_multiple_files=False
)
if uploaded_file:
    bytes_data = uploaded_file.read()
    upload_file_to_blob(STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME, "app_data/"+uploaded_file.name, bytes_data)
    st.session_state.document_url = get_blob_sas_url(STORAGE_CONNECTION_STRING, STORAGE_CONTAINER_NAME, "app_data/"+uploaded_file.name)
else:
    st.info("Please upload the PDF to continue..")
    st.stop()

st.session_state.esn = st.text_input("Enter the Engine Serial Number: ")

if "extracted_info" not in st.session_state and "retriever" not in st.session_state:
    st.session_state.extracted_info, st.session_state.extracted_text = ocr_extraction(st.session_state.document_url)
    st.session_state.retriever = configure_retriever(st.session_state.extracted_text, st.session_state.esn)
    

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
                                retriever=st.session_state.retriever, 
                                return_source_documents=True,
                                chain_type_kwargs = {"prompt": prompt})




# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)


if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("Thanks for sharing the document. Please put forward your question, and I'll do my best to help you.")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain(user_query, callbacks=[retrieval_handler, stream_handler])
        final_response = process_llm_response(response)
        st.write(final_response['result'])