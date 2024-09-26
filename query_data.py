# import argparse
from fastapi import FastAPI, Request
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context in a casual and friendly manner:

{context}

---

Answer the question in one or two sentences based on the above context: {question}
"""
response_text = {}
nlu = []

def get_response(question):
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(question, k=3)
    if len(results) == 0 or results[0][1] < 0.6:
        print(f"Unable to find matching results.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    return response_text.content