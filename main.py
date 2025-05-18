from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer
)
from llama_index.core.node_parser import MarkdownElementNodeParser
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
