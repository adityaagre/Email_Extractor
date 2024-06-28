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


def func1_text_to_email_list():
    ##################################################################################
    # making llm object
    llm = ChatOpenAI(temperature=0, verbose=False)

    ##################################################################################

    ##################################################################################
    # Takes a string and replaces all 'dot's with '.'
    # Returns strings.
    def dot_to_dot(string_conversation):
        data = string_conversation.replace('dot', '.')
        return data

    ##################################################################################

    ##################################################################################
    # Extracting interest areas from given conversation using LLM
    # So, the llm looks for areas that resemble emails and returns them.
    def area_of_interest(string_conversation):
        answer = llm.invoke(
            "Return the email ids in the following conversation. Note that emails might have '@' written as 'at the rate'. You must include such mails too. Print each email on a new line." + string_conversation)
        return answer.content

    ##################################################################################

    ##################################################################################
    ## This function is currntly not being run.
    def return_emails_with_dot(list_of_mails):
        # return list of mail ids with 'dot'.
        # First convert all mail ids to lowercase.To make checking easier.
        # Note: The presence of the letters 'dot' may not always mean '.'
        # Eg: galgadot@gmail.com
        substring_key = "dot"
        return_list = []
        for string in list_of_mails:
            if (re.search(substring_key, string)):
                return_list.append(string)
        return return_list

    ##################################################################################

    ##################################################################################
    def replace_dot_w_dot(list_of_dotted_mails):
        # replace all mails that have 'dot' in them, using '.' instead of 'dot'
        replaced_mail_list = []
        for dot_mail in list_of_dotted_mails:
            replaced_mail = dot_mail.replace('dot', '.')
            replaced_mail_list.append(replaced_mail)
        return replaced_mail_list

    ##################################################################################

    ##################################################################################
    def replace_attherate_w_attherate_statistical(list_of_all_mails):
        # replace all mails that have 'at%the%rate' in them, using '@' instead of 'at%the%rate'
        # Unlike the upper case wherein, we are taking a  list of mails which contains only 'dot' substring, here we are
        # checking for the 'at the rate's and replace them all by itself.
        # This is the statistical one. (Non LLM. We also need to write one wherein the llm shortlists the 'at the rate's
        # in case there are varying representations.)
        replaced_mail_list = []
        for atr_mail in list_of_all_mails:
            replaced_mail = atr_mail.replace('at the rate', '@')
            replaced_mail_list.append(replaced_mail)
        return replaced_mail_list

    ##################################################################################

    ##################################################################################
    def remove_spaces(list_of_spaced_mails):
        # replace all spaces ' ' in email list with '' instead of ' '.
        replaced_mail_list = []
        for spaced_mail in list_of_spaced_mails:
            replaced_mail = spaced_mail.replace(' ', '')
            replaced_mail_list.append(replaced_mail)
        return replaced_mail_list

    ##################################################################################

    ##################################################################################
    def choose_best_using_llm_call_for_each_singular_email_for_atr_transform(string_single_mail_replaced,
                                                                             string_single_mail_original):
        # This function will take two email ids and choose the one that is most approporiate.
        # To decide if 'at the rate' to '@' replacement makes it better, or was the orignal one better.

        prompt_seq1 = [
            "You will now be provided with two forms of the same email id. Analyse both carefully and decide which email seems most appropriate. ",
            "Go through both these emails and return the email you think is most appropriate. Do not answer in a senternce. Simply output a singular email id."]
        # answer = llm.invoke("You will now be provided with two forms of the same email id. Analyse both carefully and decide which email seems most appropriate. " + string_single_mail_original + string_single_mail_replaced + "Go through both these emails and return the email you think is most appropriate. Do not answer in a senternce. Simply output a singular email id.")
        answer = llm.invoke(prompt_seq1[0] + string_single_mail_original + string_single_mail_replaced + prompt_seq1[1])
        return answer.content

    ##################################################################################

    ##################################################################################
    def choose_best_using_llm_call_for_each_singular_email_for_dot_transform(string_single_mail_replaced,
                                                                             string_single_mail_original):
        # This function will take two email ids and choose the one that is most approporiate.
        # To decide if 'dot' to '.' replacement makes it better, or was the orignal one better.
        # Note: The presence of the letters 'dot' may not always mean '.'
        # Eg: galgadot@gmail.com

        prompt_seq1 = [
            "You will now be provided with two forms of the same email id. Analyse both carefully and decide which email seems most appropriate. ",
            "Go through both these emails and return the email you think is most appropriate. Do not answer in a senternce. Simply output a singular email id."]
        prompt_seq2 = [
            "You will now be provided with two forms of the same email id. Analyse both carefully and decide which email seems most appropriate. Note that both of thse emails will be containing examples wherein email id pairs where 'dot' is replaced by '.'. For example, 'abcdotxyz@gmaildotcom' and 'abc.xyz@gmail.com'. Here the second mail is more appropriate. However, in the following pair: 'galgadot@gmail.com' and 'galga.@gmail.com'. Now in this case, the 'dot' in 'galgadot' is a part of the person's name. Therefore replacing it with 'dot would not be correct'. Therefore you need to understand by thinking carefully as of which email id seems most appropriate. Here are the emails : ",
            "Go through both these emails and return the email you think is most appropriate. Do not answer in a senternce. Simply output a singular email id."]
        # answer = llm.invoke("You will now be provided with two forms of the same email id. Analyse both carefully and decide which email seems most appropriate. " + string_single_mail_original + string_single_mail_replaced + "Go through both these emails and return the email you think is most appropriate. Do not answer in a senternce. Simply output a singular email id.")
        # answer = llm.invoke(prompt_seq1[0] + string_single_mail_original + string_single_mail_replaced + prompt_seq1[1])
        answer = llm.invoke(prompt_seq2[0] + string_single_mail_original + string_single_mail_replaced + prompt_seq2[1])
        return answer.content

    ##################################################################################

    ##################################################################################
    # Calling environment
    with open('data.txt', 'r') as file:
        data = file.read()

    string_convo = """Alice: Hey Bob, did you get the email I sent about the project update? I sent it from alice.johnson@example.com.

    Bob: Hi Alice, I didn't see it in my inbox. Can you resend it to bob.D.smith@example.com? Maybe it got lost in the shuffle.

    Charlie: Hey Alice and Bob, I'm here too. Can you CC me on the project update? My email is KK.charlie.brown@example.net.

    Alice: Sure thing, Charlie. I'll resend the email and CC you. By the way, has anyone heard from Dave? His email is david.williams@example.org, but I haven't gotten any response.

    Bob: I haven't heard from him either. Maybe we should try his alternate email: dave.w@example.com.

    Charlie: Good idea. Also, if you need to reach Emily, her email is s.emily.clark@example.edu. She's been handling the client side of things.

    Alice: Thanks, Charlie. I'll loop her in. Just to confirm, should I also include Fiona? Her email is fiona.k.martin@example.co.uk, right?

    Bob: Yes, please include Fiona. And don't forget to send a copy to the team lead at teamlead@example.biz.

    Charlie: Speaking of the team lead, do we need to update the team list? I think we've added a new member, Greg. His email is greg.jones@example.co.

    Alice: Got it. I'll make sure to update the list. Oh, and for the upcoming meeting, can you guys send the agenda to the shared mailbox? It's meeting.agenda@example.com.

    Bob: Sure thing, Alice. And if anyone needs IT support, the help desk email is support@example.tech.

    Charlie: Thanks for the reminder, Bob. I've had a few tech issues lately. I'll reach out to them. Also, Alice, can you share the project documentation with documentation@example.io?

    Alice: Will do, Charlie. I'll send everything out shortly. Let's make sure we have all the emails in one place. It's been a bit scattered lately.

    Bob: Agreed. Thanks for organizing this, Alice. It makes everything much easier.

    Charlie: Definitely. Thanks, Alice!




    hello@gmail.com

    Alice: Hey Bob, did you get the email I sent about the project update? I sent it from alicedotjohnson at the rate exampledotcom.

    Bob: Hi Alice, I didn't see it in my inbox. Can you resend it to bobemdotenDdotsmith at the rate exampledotcom? Maybe it got lost in the shuffle.

    Charlie: Hey Alice and Bob, I'm here too. Can you CC me on the project update? My email is KKdotcharliedotbrown at the rate exampledotnet.

    Alice: Sure thing, Charlie. I'll resend the email and CC you. By the way, has anyone heard from Dave? His email is daviddotwilliams at the rate exampledotorg, but I haven't gotten any response.

    Bob: I haven't heard from him either. Maybe we should try his alternate email: davedotw at the rate exampledotcom.

    Charlie: Good idea. Also, if you need to reach Emily, her email is sdotemilydotclark at the rate exampledotedu. She's been handling the client side of things.

    Alice: Thanks, Charlie. I'll loop her in. Just to confirm, should I also include Fiona? Her email is fionadotkdotmartin at the rate exampledotcodotuk, right?

    Bob: Yes, please include Fiona. And don't forget to send a copy to the team lead at teamlead at the rate exampledotbiz.

    Charlie: Speaking of the team lead, do we need to update the team list? I think we've added a new member, Greg. His email is gregdotjones at the rate exampledotco.

    Alice: Got it. I'll make sure to update the list. Oh, and for the upcoming meeting, can you guys send the agenda to the shared mailbox? It's meetingdotagenda at the rate exampledotcom.

    Bob: Sure thing, Alice. And if anyone needs IT support, the help desk email is support at the rate exampledottech.

    Charlie: Thanks for the reminder, Bob. I've had a few tech issues lately. I'll reach out to them. Also, Alice, can you share the project documentation with documentation at the rate exampledotio?

    Alice: Will do, Charlie. I'll send everything out shortly. Let's make sure we have all the emails in one place. It's been a bit scattered lately.

    Bob: Agreed. Thanks for organizing this, Alice. It makes everything much easier.

    Charlie: Definitely. Thanks, Alice!"""
    string_convo2 = """
    Alice: Hey Bob, did you get the email I sent about the project update? I sent it from alicedotjohnson at the rate exampledotcom.

    Bob: Hi Alice, I didn't see it in my inbox. Can you resend it to bobemdotenDdotsmith at the rate exampledotcom? Maybe it got lost in the shuffle.

    Charlie: Hey Alice and Bob, I'm here too. Can you CC me on the project update? My email is KKdotcharliedotbrown at the rate exampledotnet.

    Alice: Sure thing, Charlie. I'll resend the email and CC you. By the way, has anyone heard from Dave? His email is daviddotwilliams at the rate exampledotorg, but I haven't gotten any response.

    Bob: I haven't heard from him either. Maybe we should try his alternate email: davedotw at the rate exampledotcom.

    Charlie: Good idea. Also, if you need to reach Emily, her email is sdotemilydotclark at the rate exampledotedu. She's been handling the client side of things.

    Alice: Thanks, Charlie. I'll loop her in. Just to confirm, should I also include Fiona? Her email is fionadotkdotmartin at the rate exampledotcodotuk, right?

    Bob: Yes, please include Fiona. And don't forget to send a copy to the team lead at teamlead at the rate exampledotbiz.

    Charlie: Speaking of the team lead, do we need to update the team list? I think we've added a new member, Greg. His email is gregdotjones at the rate exampledotco.

    Alice: Got it. I'll make sure to update the list. Oh, and for the upcoming meeting, can you guys send the agenda to the shared mailbox? It's meetingdotagenda at the rate exampledotcom.

    Bob: Sure thing, Alice. And if anyone needs IT support, the help desk email is support at the rate exampledottech.

    Charlie: Thanks for the reminder, Bob. I've had a few tech issues lately. I'll reach out to them. Also, Alice, can you share the project documentation with documentation at the rate exampledotio?

    Alice: Will do, Charlie. I'll send everything out shortly. Let's make sure we have all the emails in one place. It's been a bit scattered lately.

    Bob: Agreed. Thanks for organizing this, Alice. It makes everything much easier.

    Charlie: Definitely. Thanks, Alice!"""
    provisional_emails_llm_response = area_of_interest(data)
    first_mail_list = re.split('\n', provisional_emails_llm_response)
    atr_changed_mails = replace_attherate_w_attherate_statistical(first_mail_list)
    # dotted_mails = return_emails_with_dot(atr_changed_mails)
    replaced_dotted_mails = replace_dot_w_dot(atr_changed_mails)
    chosen_mails_amongst_dotted_and_dot_replaced = []
    for i in range(len(atr_changed_mails)):
        chosen_mail = choose_best_using_llm_call_for_each_singular_email_for_dot_transform(atr_changed_mails[i],
                                                                                           replaced_dotted_mails[i])
        chosen_mails_amongst_dotted_and_dot_replaced.append(chosen_mail)
    unspaced_mails = remove_spaces(chosen_mails_amongst_dotted_and_dot_replaced)
    return unspaced_mails

    # email_progression = {
    #     'First_mail_list': first_mail_list,
    #     'At_the_rate_changed_to @': atr_changed_mails,
    #     '"Dot" replaced with "."': replaced_dotted_mails,
    #     'Mails chosen among "dot" and "dot replaced by ."': chosen_mails_amongst_dotted_and_dot_replaced,
    #     'Spaces removed': unspaced_mails
    # }
    #
    # display_progress = pd.DataFrame(email_progression)
    # print(display_progress)
    #
    # with open('display_progress_dictionaries.txt', 'a') as f:
    #     f.write(str(email_progression))
    #     f.write("\n")
    #     f.close()

    # print(first_mail_list, "\n", atr_changed_mails, "\n", dotted_mails, "\n", replaced_dotted_mails, "\n", unspaced_mails)
    ##################################################################################

print(func1_text_to_email_list())

