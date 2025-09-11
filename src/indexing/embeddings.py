import os
import pickle
import sys

import pandas as pd

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.helper import Paths

def load_preprocess_data() -> pd.DataFrame:

    columns_rename = {
        "BrandName": "Brand Name",
        "Deatils": "Product Details",
        "Sizes": "Available Sizes",
        "SellPrice": "Product Price",
        "Category": "Product Category"
    }

    valid_cols = [
        "Brand Name",
        "Product Details", 
        "Available Sizes", 
        "Product Price",
        "Product Category"
    ]

    df = pd.read_csv(f"{Paths.DATA_DIR}/FashionDataset.csv")

    df = df.rename(columns={k: v for k, v in columns_rename.items() if k in df.columns})
    df = df[[col for col in valid_cols if col in df.columns]]

    df.dropna(inplace=True)

    df.to_csv(f"{Paths.DATA_DIR}/dataset_processed.csv")

    return df

def generate_documents() -> list:

    if os.path.exists(f"{Paths.DATA_DIR}/dataset_processed.csv"):
        loader = CSVLoader(f"{Paths.DATA_DIR}/dataset_processed.csv")
        documents = loader.load()

        return documents
    
    else:
        return "Dataset not found"

def initialize_embeddings() -> HuggingFaceEmbeddings:

    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        return embeddings
    
    except Exception as e:
        raise e

def create_faiss_index(embeddings, documents) -> None:

    try: 
        faiss_index = FAISS.from_documents(documents, embeddings)
        faiss_index.save_local(f"{Paths.INDEX_DIR}/faiss_index.faiss")
    
    except Exception as e: 
        raise e

def create_chroma_index(embeddings, documents) -> None:

    try:
        vector_store = Chroma(
            collection_name="product_collection",
            embedding_function=embeddings,
            persist_directory=f"{Paths.INDEX_DIR}/chroma_index"
        )

        batches = [documents[i:i+5461] for i in range(0, len(documents), 5461)]
        for batch in batches:
            vector_store.add_documents(batch)

    except Exception as e:
        raise e

def create_bm25_index(documents) -> None:

    try:
        os.makedirs(os.path.dirname(f"{Paths.INDEX_DIR}/bm25_index.pkl"), exist_ok=True)

        bm25_index = BM25Retriever.from_documents(documents)

        with open(f"{Paths.INDEX_DIR}/bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_index, f)

    except Exception as e:
        raise e


def pipeline() -> None: 

    try:
        df = load_preprocess_data()
        documents = generate_documents()
        embeddings = initialize_embeddings()

        create_faiss_index(embeddings, documents)
        create_chroma_index(embeddings, documents)
        create_bm25_index(documents)

    except Exception as e:
        raise e
    

if __name__ == "__main__": 
    pipeline()

