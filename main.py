from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

load_dotenv()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Global settings (still will pass embed_model as param)
Settings.embed_model = embed_model
Settings.llm = OpenAI(model="gpt-4o-mini")

# Process the docs
document = SimpleDirectoryReader(input_files=["data/docs.md"]).load_data()
text_splitter = MarkdownNodeParser()
Settings.text_splitter = text_splitter

# Vector database setup
CHROMA_DB_PATH = "./chroma_db"
COLLECTION = "neatqueue"

db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = db.get_or_create_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if len(chroma_collection.get()["ids"]) == 0:
    index = VectorStoreIndex.from_documents(
        document,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[text_splitter],
    )
else:
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
