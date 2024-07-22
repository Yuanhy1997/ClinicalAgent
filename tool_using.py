

import datetime
import json
import openai
import os
import pinecone
import re
from tqdm.auto import tqdm
from typing import List, Union
import zipfile

# Langchain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# LLM wrapper
from langchain_huggingface import HuggingFaceEndpoint
# Conversational memory
from langchain.memory import ConversationBufferWindowMemory
# Embeddings and vectorstore
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough

from langchain import hub


model_name = "./hf_models/sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
os.environ['PINECONE_API_KEY'] = 'b5dd19d6-7568-4224-bd13-d9acfe6653e0'
index_name = "test"
loader = JSONLoader(
    file_path= "./database/mimiciii.json",
    jq_schema='.data[].note',
    text_content=False)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
vectorstore = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embedding
)
retriever = vectorstore.as_retriever() 
query = "the paitent has allergies related to codeine."  
# result = vectorstore.similarity_search(  
#     query,  # our search query  
#     k=1  # return 3 most relevant docs  
# )  
# print(result)


prompt = hub.pull("rlm/rag-prompt")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_path = "./hf_models/microsoft/Phi-3-mini-4k-instruct/"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
#max_length has typically been deprecated for max_new_tokens 
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, model_kwargs={"temperature":0}
)
llm = HuggingFacePipeline(pipeline=pipe)


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print(rag_chain.invoke(query))
# # Define a list of tools
# tools = [
#     Tool(
#         name = "Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events"
#     )
# ]


# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )
# from langchain_core.messages import (
#     HumanMessage,
#     SystemMessage,
# )
# from langchain_huggingface import ChatHuggingFace

# messages = [
#     SystemMessage(content="You're a helpful assistant"),
#     HumanMessage(
#         content="What happens when an unstoppable force meets an immovable object?"
#     ),
# ]

# chat_model = ChatHuggingFace(llm=llm)

# res = chat_model.invoke(messages)
# print(res.content)
