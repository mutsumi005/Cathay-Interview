from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("📄 載入 PDF 條款...")

loader = PyPDFLoader("海外旅行不便險條款.pdf")
documents = loader.load()

print(f"📑 條款頁數：{len(documents)}")

# 切條款（保險很適合小 chunk）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

docs = text_splitter.split_documents(documents)

print(f"✂️ 切分後條款數：{len(docs)}")

# 中文 embedding
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

print("🧠 建立 FAISS 索引中...")

vectorstore = FAISS.from_documents(docs, embedding)

vectorstore.save_local("insurance_faiss")

print("✅ 索引建立完成！儲存在 insurance_faiss/")
