import os
import pickle
import sys

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.helper import Paths

def load_faiss_index() -> FAISS:

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    try: 
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = FAISS.load_local(f"{Paths.INDEX_DIR}/faiss_index.faiss", embeddings_model, allow_dangerous_deserialization=True)
    except Exception as e:
        raise e
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


def load_bm25_index() -> object:

    try: 
        with open(f"{Paths.INDEX_DIR}/bm25_index.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        return bm25_retriever
    
    except Exception as e:
        raise e


def create_ensemble(retrievers) -> EnsembleRetriever:

    return EnsembleRetriever(retrievers=retrievers, weights=[0.5, 0.5], top_k=5)


def create_cross_encoder_reranker(ensemble_retriever) -> None:

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=model, top_n=3)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)


def save_cross_encoder_reranker(cross_encoder_reranker) -> None:

    try: 
        with open(f"{Paths.INDEX_DIR}/cross_encoder_reranker.pkl", "wb") as f:
            pickle.dump(cross_encoder_reranker, f)
    
    except Exception as e:
        raise e
    

def retrieval_flow() -> None:

    try: 

        faiss_retriever = load_faiss_index()
        bm25_retriever = load_bm25_index()

        retrievers = [faiss_retriever, bm25_retriever]

        ensemble_retriever = create_ensemble(retrievers)
        cross_encoder_reranker = create_cross_encoder_reranker(ensemble_retriever)

        save_cross_encoder_reranker(cross_encoder_reranker)

    except Exception as e:

        raise e
    
if __name__ == "__main__": 
    retrieval_flow()