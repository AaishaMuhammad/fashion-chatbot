import os
import sys

from langchain.globals import set_llm_cache
from langchain.schema.output_parser import StrOutputParser
from langchain_community.cache import InMemoryCache
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.recommender_graph.state_graph import State

def create_rag_template():

    prompt_template = """You are an intelligent shopping assistant that helps users find the best products based on their query.

    The user is looking for products related to: **{query}**.

    Here are some available products:
    {docs}

    Please recommend the best products in a friendly, conversational tone. Consider the following:
    - **Match with the user's preferences** (e.g., price, size, brand).
    - **High user ratings and popularity**.
    - **Relevance to the user's intent**.

    **Respond in natural language as if you were personally assisting the user.**
    
    Example response:
    "Based on your request for {query}, here are some great options: 
    1. [Product A] - A great choice because...
    2. [Product B] - This one stands out due to...
    
    Let me know if you need more details or alternatives!"
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["docs", "query"])
    return prompt

def build_rag_chain():

    # model_name = "llama3.2:3b"
    model_name = "gemma3n:e2b"
    set_llm_cache(InMemoryCache())

    llm = ChatOllama(
        model=model_name,
        temperature=0,
        max_tokens=100,
        cache=True
    )

    prompt = create_rag_template()
    parser = StrOutputParser()

    rag_chain = (
        RunnableParallel(
            {
                "docs": RunnableLambda(lambda x: x["docs"]),
                "query": RunnableLambda(lambda x: x["query"])
            }
        )
        | prompt
        | llm
        | parser
    )

    return rag_chain

def rag_recommender(state) -> State:

    rag_chain = build_rag_chain()
    query = state["query"]
    docs = state["products"]

    output = rag_chain.invoke({"docs": docs, "query": query})
    state["recommendation"] = output

    return state