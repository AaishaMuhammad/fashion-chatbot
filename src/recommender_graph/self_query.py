import os
import sys

from functools import lru_cache
from typing import List

from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, wait_fixed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.recommender_graph.chroma_translator import CustomChromaTranslator
from src.recommender_graph.state_graph import State



ATTRIBUTE_INFO = [
    {
        "name": "Product Details",
        "description": "Details about the product",
    },
    {
        "name": "Brand Name",
        "description": "Name of the brand",
    },
    {
        "name": "Available Sizes",
        "description": (
            "Sizes available for the product (stored as a comma-separated string, e.g., 'small, medium, large'). "
            "Use the `like` operator to check if a size is included. "
            'Example: `like("Available Sizes", "xl")` to find products that have XL in their size options.'
        ),
    },
    {
        "name": "Product Price",
        "description": "Price of the product. Use `lt`, `lte`, `gt`, or `gte` for filtering.",
    },
]

DOC_CONTENT = "A detailed description of an e-commerce product, including its features, benefits, and specifications."


def get_metadata_info():
    return ATTRIBUTE_INFO, DOC_CONTENT

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@lru_cache(maxsize=1)
def initialize_embeddings_model() -> HuggingFaceEmbeddings:

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    try: 
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    
    except Exception as e:
        raise e
    
def load_chroma_index(embeddings) -> Chroma:

    try:
        vectorstore = Chroma(
            collection_name="product_collection",
            embedding_function=embeddings,
            persist_directory="indexes/chroma_index"
        )

        return vectorstore
    
    except Exception as e:
        raise e
    

def build_self_query_chain(vectorstore) -> RunnableLambda:

    llm = ChatOllama(
        model="gemma3n:e2b",
        temperature=0
    )

    attribute_info, doc_contents = get_metadata_info()

    query_constructor = load_query_constructor_runnable(
        llm=llm,
        document_contents=doc_contents,
        attribute_info=attribute_info
    )

    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        verbose=True,
        structured_query_translator=CustomChromaTranslator()
    )

    self_query_chain = RunnableLambda(lambda inputs: retriever.invoke(inputs["query"]))
    return self_query_chain


def self_query_retriever(state) -> State:

    embeddings = initialize_embeddings_model()
    chroma_index = load_chroma_index(embeddings)
    self_query_chain = build_self_query_chain(chroma_index)

    def format_docs(docs):

        return "\n\n".join([f"- {doc.page_content}" for doc in docs])
    
    query = state["query"]

    results = self_query_chain.invoke({"query": query})

    if len(results) == 0:
        state["self_query_state"] = "empty"

    else: 
        state["self_query_state"] = "success"
        state["products"] = format_docs(results)

    return state


