import os
import sys
import pickle

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.recommender_graph.state_graph import State

def load_cross_encoder_model() -> HuggingFaceEmbeddings:

    try:
        with open("./indexes/cross_encoder_reranker.pkl", "rb") as f: 
            cross_encoder = pickle.load(f)
        
        return cross_encoder
    
    except Exception as e:

        raise e
    
def build_ranker(query): 

    cross_encoder = load_cross_encoder_model()

    def format_docs(docs): 
        return "\n\n".join([f"- {doc.page_content}" for doc in docs])

    product_docs = cross_encoder.invoke(query)
    products = format_docs(product_docs)

    return products


def ranker_node(state) -> State:

    query = state["query"]
    product_list = build_ranker(query)
    state["products"] = product_list

    return state