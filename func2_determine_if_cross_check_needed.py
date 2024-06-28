## imports
import re
import pandas as pd
import ast
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
#from langchain_community.vectorstores import FAISS
import openai
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import Tool
#import pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import OpenAI, ConversationChain
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
#from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool
from signal import signal, SIGALRM, alarm
#from flask import Flask, render_template, request, jsonify

##################################################################################
## Environment veriables, API Keys, etc
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
##################################################################################

def is_email_legible_llm_version(transcript, email_id):
    ##################################################################################
    # making llm object
    llm = ChatOpenAI(temperature=0, verbose=False)

    ##################################################################################

    answer = llm.invoke("Given the following call transcript, return if the email seems to be correctly extracted or if the user must be asked again to clarify their email. Here is the transcript: " + transcript + "This is the inferred email: " + email_id + "Now, if you think email id is correctly extracted and there seems to be no room for any discrepency, return your answer as a singular word, 'True'. If you think user needs to be asked again to clarify what the said, return a singular word, 'False'.")
    return answer.content
