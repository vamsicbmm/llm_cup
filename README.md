# llm_cup
This repo contains the code for FinTwin SkyBot

## Inspiration
The existence of an aviation asset is dependent on the physical paperwork. This creates a huge challenge for manually analyzing huge amounts of data. Digitizing the physical paperwork can reduce a lot of manual work and generate value within seconds. The FinTwin Skybot, powered by LLM, can input multiple PDFs and provide crucial insights from the documents.
## What it does
Skybot takes in the asset information as a PDF and provides insights based on user queries. It can perform complex computations, such as shop visit forecasts. A shop visit is when the engine is taken to maintenance. Based on the details provided in the PDF, Skybot can provide an accurate forecast. The Skybot extracts the Life Limited Part (LLP) information from the PDF with the help of the OCR tool and then calls the shop visit forecaster tool in order to forecast when the next shop visits can happen in the next 50000 cycles. That means the Skybot will provide shop visit intervals in the next 50000 cycles. Similarly, we can add as many different analytic functionalities using agents by using complex functions and some amazing prompting.
## How we built it
We used the RAG (Retrieval Augumentation Generation) system to retrieve the relevant information based on the user query. Then pass it to LLM, which will trigger agents based on the type of analysis required by the users. We have built different agents, like OCR agents to extract information from PDFs, forecasting agents, and cost-computing agents. Based on the analytical query passed by the user, the LLM will trigger these agents. To improve the retrival system, we fine-tuned the embeddings using aviation-specific documents and other unrelated documents to provide negative samples in the training process. It improved the retrieval accuracy by 10%. For agent output monitoring, we are using Langsmith to make sure the LLM outputs are ethical and not harmful. 

## Challenges we ran into
- The pre-trained embedding for RAG was not retrieving the write information based on the user query.
- Managing the prompts for the right orchestration using agents was difficult. 
- Building an agent that can connect the outputs of different tools to produce the final output was challenging due to the inherent nature of LLMs ability to generate solutions by themselves.
- Fine-tuning the LLM based on domain knowledge was difficult due to a lack of digitized domain data and computational challenges in setting up the platform.
- Tracing the output from the user query and debugging it to test whether the LLM is providing the desired output or not.

## Accomplishments that we're proud of
- We successfully fine-tuned the embeddings for our use case and improved the retrieval accuracy.  
- We have built an LLM-based agent that can successfully predict the next shop visit for an engine by using intelligent tools to extract necessary details from its minipack and use them to forecast the next shop visit. 
- We have made use of tracing tools such as LangSmith to debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework.
- The fine-tuning setup is done. Very minimal data is taken and fine-tuned using LoRA (Low Rank Adapter) with llama-7B as the base model. 

## What we learned
- We have learned a great deal about building various kinds of agents to get an LLM to understand the user's requirements and invoke the right kind of tools to perform the tasks. 
- The accuracy of the fine-tuning of word embeddings was totally dependent on the variety of data required to tune. similar examples as well as dissimilar examples.

## What's next for Fintwin Skybot?
- To make a better-fine-tuned aviation-specific LLM, we need to collect a large amount of text data and follow the fine-tuning process. This is the very first future work for this Asset Analyzer
- Adding an SME feedback layer to a deployed system, SMEs can change the prompt to provide control to the users.
- Using the Azure semantic kernel to orchestrate all the tools using the semantic layer.
- Make the model capable of generating more financial-related insights required in the aviation industry.
- Connecting multiple databases to agents for better results.
- The shop visit forecast tool will be modified with the latest algorithm to make better predictions.

## How to run the code? 

**Step1**: Run ``requirements.txt``


**Step2**: Run ```streamlit run app.py```
