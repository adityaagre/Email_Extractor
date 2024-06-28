def extract_details(transcript):
    ##imports
    import re
    import pandas as pd
    import ast
    # from langchain.chat_models import ChatOpenAI
    # from langchain import PromptTemplate, LLMChain
    # from langchain_community.vectorstores import FAISS
    import openai
    import os
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    from langchain.agents import Tool
    # import pypdf
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import UnstructuredHTMLLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain import OpenAI, ConversationChain
    from langchain.agents import initialize_agent
    from langchain.chains.conversation.memory import ConversationBufferWindowMemory
    # from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool
    from signal import signal, SIGALRM, alarm
    # from flask import Flask, render_template, request, jsonify

    ##################################################################################
    ##################################################################################
    ## Environment veriables, API Keys, etc
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    ##################################################################################

    ##################################################################################
    # making llm object
    llm = ChatOpenAI(model = "gpt-4o", temperature=0, verbose=False)

    ##################################################################################
    prompt1 = """
    Go through the following conversational transcript : """
    prompt2 = """
    Now return a json of the following nature: Note: Do not answer in complete sentences in any case. In case of the fields wherein options are , pick any one. Write only one. Options are seperated by '|'
    {
    'Full Name': 'Danielle Cavalier', 
    'Address': 'Sample address' | 'None', 
    'Phone Number of Customer': '203-555-1234' | 'None', 
    'Email ID of Customer': 'Thealarmgroup@gmail.com' | 'None', 
    'POA?': 'Yes'|'No.'|'Unknown', 
    'POA - Relation. ': 'Yes'|'No.'|'Unknown', 
    'New Medical Pendant OR Existing?': 'Yes'|'No.'|'Unknown', 
    'How much paying per month?': 'amount'|'0', 
    'Is the customer in contract': 'Yes'|'No.'|'Unknown', 
    'Does the Customer Qualify?': 'Yes'|'No.'|'Unknown', 
    'Processing Date': 'DD/MM/YYYY'|'None', 
    'Recurring Date': 'DD/MM/YYYY'|'None', 
    'Other Promos?': 'Home Security' | 'Life Insurance'| 'Solar' | 'AT&T Services' | 'None'
    }
    """
    answer = llm.invoke(prompt1 + transcript + prompt2)
    return answer.content


# transcript = """Agent: Hi, My name is Ashley with Optimum Life Care Services, How are you today?
# User: I'm doing great.
# Agent: That's wonderful to hear! I'm reaching out to people in your area today to verify if you are a senior or if there is a senior living with you, or any family members that need medical attention?
# User: Yes.
# Agent: Fantastic! As I mentioned, my name is Ashley, and the reason for my call is to inform you that you have been selected to receive a Life Alert-like medical pendant at no cost! This device is valued at $400, but through this promotion, it is completely free!
#
# Do you know how a life alert or medical pendant works?
# User: No.
# Agent: No worries! Let me explain. Our medical pendant comes as a necklace or wristband with a panic button in the middle. It has a live monitoring station available 24/7 for emergencies. You can wear it outside, even while shopping. It includes GPS and AT&T, 4G, LTE, and is waterproof, so you can wear it while bathing. Plus, it has a speaker and microphone for direct communication with an agent during emergencies.
#
# Does that sound like something that could be useful for you or your loved one?
# User: I'm sorry.
# Agent: No problem at all! Would you be
# User: Repeat that again.
# Agent: interested in Home Security, Life Insurance, Solar, or AT&T Services instead?
# User: Uh, no.
# User: I am interested in medical pendant. Can you continue with that?
# Agent: Of course! To ensure you qualify, I need to ask you a few questions:
#
# 1. Do you currently have or use a medical pendant?
# User: No.
# Agent: Got it. Do you have a son, daughter, or Power of Attorney to help you make decisions?
# User: No.
# Agent: Alright, no problem. Are you currently under a contract for a medical pendant?
# User: No.
# Agent: Great! Before we proceed, for legal purposes, you are giving us permission to contact you even if your number is a wireless number or has been listed on a Do Not Call list. Is that okay?
# User: Yes.
# Agent: Ok, great! It looks like you qualify!
#
# Now, allow me to explain the benefits of our services. Optimum Life Care is the top brand on the market, providing exact location pinpointing in less than 30 seconds. With GPS and AT&T, 4G, LTE, no landline is required. The system includes fall detection and a speaker/microphone for emergencies.
#
# Which would you prefer, a necklace or a bracelet?
# User: Necklace.
# Agent: Excellent choice! Normally, the equipment costs $400, but today it's free! You only pay a one-time fee of $199 plus a monitoring fee at a discounted rate of $39.99 per month. Sound good?
# User: Yeah.
# Agent: Perfect! Let me confirm your information to send out your package today.
#
# Can I have your full name, please?
# User: Danielle Cavalier.
# Agent: Thank you, Danielle Cavalier. Let me spell that out to make sure I have it right: D-A-N-I-E-L-L-E C-A-V-A-L-I-E-R. Is that correct?
# User: Yes.
# Agent: Great! Could you please provide your phone number?
# User: 203 5551234.
# Agent: Thank you! Let me confirm that: 203-555-1234. Is that correct?
# User: Yes.
# Agent: Perfect! Could I have your email address, please?
# User: Thealarmgroup@gmail.com.
# Agent: Thank you! Let me confirm that: Thealarmgroup@gmail.com. That's T-H-E-A-L-A-R-M-G-R-O-U-P @ G-M-A-I-L . C-O-M. Is that correct?
# User: Yes.
# Agent: Great! Now, could you please provide your date of birth?"""

f= open("transcript.txt", "r")
transcript = f.read()

print(extract_details(transcript))