

import google.generativeai as genai
from datasets import load_dataset
import os
from pinecone import Pinecone, ServerlessSpec
import time
pc= Pinecone(api_key=os.getenv("PINECONE_API_KEY"),environment="us-west1-gcp")

dataset=load_dataset("jamescalam/llama-2-arxiv-papers-chunked",
                     split="train")
index_name="llama-2-rag"

if index_name not in pc.list_indexes():
    
    pc.create_index(index_name,dimension=1536,metric='cosine',spec=ServerlessSpec(region="us-west1-gcp"))
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
index=pc.index(index_name)
index.describe_index_stats()

print(index)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    
    
# model= genai.GenerativeModel('gemini-pro')
# response=model.generate_content("what is so special about Llama 2 ? ")
# print(response.text)
# print(response.prompt_feedback)