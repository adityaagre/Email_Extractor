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

##################################################################################
# making llm object
llm = ChatOpenAI(temperature=0, verbose=False)

ans = llm.invoke("Generate 50 sample email ids. return them as comma seperated values. output only the emails with commas like 'amit_kumar92@gmail.com, alka.yagnik@yahoo.co.uk, therealsteven@hotmail.com'")
print(ans)


