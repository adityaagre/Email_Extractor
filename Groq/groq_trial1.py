# (python) (base) adityaagre@ADITYAs-MacBook-Air-2 ~ % export GROQ_API_KEY=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
# (python) (base) adityaagre@ADITYAs-MacBook-Air-2 ~ % pip install groq



import os
import groq
from dotenv import load_dotenv

##################################################################################
## Environment veriables, API Keys, etc
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
##################################################################################


from groq import Groq

client = Groq(
    # api_key=os.environ.get("GROQ_API_KEY"),
    api_key = GROQ_API_KEY,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Write 10 sample email ids.",
        }
    ],
    model="llama3-70b-8192",
)
ans = chat_completion.choices[0].message
print(ans.content)

import requests
import os

api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())