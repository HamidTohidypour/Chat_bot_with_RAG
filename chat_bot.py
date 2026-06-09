from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ----------------------------
# 1. Load docs
# ----------------------------
loader = TextLoader("data.txt")
docs = loader.load()

# ----------------------------
# 2. Split
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ----------------------------
# 3. Embeddings
# ----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ----------------------------
# 4. Vector DB
# ----------------------------
vector_db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vector_db.as_retriever()

# ----------------------------
# 5. LLM
# ----------------------------
llm = Ollama(model="llama3.1")

# ----------------------------
# 6. Prompt
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using the context below.

Context:
{context}

Question:
{question}
""")

# ----------------------------
# 7. Format docs
# ----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ----------------------------
# 8. RAG chain (LCEL)
# ----------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# ----------------------------
# 9. Chat loop
# ----------------------------
while True:
    q = input("You: ")
    if q == "exit":
        break
    print("Bot:", rag_chain.invoke(q))