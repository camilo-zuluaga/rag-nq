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

template = """
You are a knowledgeable and precise assistant specialized in question-answering commands and guidance for a discord bot,
particularly a bot that can create an entire queue system in discord called NeatQueue.
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:

Comprehension and Accuracy: Carefully read and comprehend the provided context from the research paper to ensure accuracy in your response.
Conciseness: Deliver the answer in no more than five sentences, ensuring it is concise and directly addresses the question.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't have this information."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.
Knowledge: Assume no prior knowledge, please dont answer commands that dont exist on your context.

Remember if no context is provided please say you don't know the answer
Here is the question and context for you to work with:

\nQuestion: {question} \nContext: {context} \nAnswer:"""

prompt_tmpl = PromptTemplate(
    template=template,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)
