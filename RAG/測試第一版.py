import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# ======================
# 1. Embedding (保持輕量)
# ======================
print("📦 初始化向量模型...")
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# ======================
# 2. 載入 FAISS 索引
# ======================
print("✅ 載入已存在索引...")
vectorstore = FAISS.load_local(
    "insurance_faiss",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 再次減少 K 值以節省顯存

# ======================
# 3. 載入 Qwen2-1.5B (保證能動的版本)
# ======================
# 改用 1.5B 模型，顯存佔用約 1.5GB ~ 2GB
# ======================

# ======================
# ======================
# 3. 載入 Qwen2-1.5B (原生 FP16 版)
# ======================
model_id = "Qwen/Qwen2-1.5B-Instruct" 

print(f"🔥 正在以原生 FP16 模式載入 {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 核心修改：不使用 quantization_config，直接用 torch_dtype
# 1.5B 模型在 float16 模式下僅佔約 3.2GB 顯存，你的 6GB 顯卡綽綽有餘！
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    device_map="auto",          # 原生 FP16 模型支援自動分配
    trust_remote_code=True
)

# pipeline 保持不變
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=pipe)
# ======================
# 4. Prompt 與 Chain
# ======================
template = """你是一位保險專業顧問，請根據提供的「海外旅行不便險條款」精確回答問題。

條款內容：
{context}

問題：{question}
答案："""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 注意：確保你的 langchain 是新版，否則請維持原本的匯入方式
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ======================
# 5. Gradio 介面函式
# ======================
def ask_insurance(question):
    if not question.strip(): return "請輸入問題", ""
    try:
        # 使用 invoke 避免舊版警告
        result = qa.invoke({"query": question})
        answer = result["result"]
        # 處理模型可能的冗長輸出，只抓答案部分
        if "答案：" in answer:
            answer = answer.split("答案：")[-1].strip()
            
        sources = "\n\n".join([f"📄 條款摘錄：\n{doc.page_content[:300]}" for doc in result["source_documents"]])
        return answer, sources
    except Exception as e:
        return f"發生錯誤：{str(e)}", ""

# ======================
# 6. 啟動介面
# ======================
with gr.Blocks(title="保險 RAG 系統") as demo:
    gr.Markdown("# 🧳 輕量版保險條款問答系統\n針對 6GB 顯存優化 (Qwen2-1.5B)")
    with gr.Row():
        with gr.Column(scale=1):
            q = gr.Textbox(label="請輸入您的問題", placeholder="例如：班機延誤怎麼賠？")
            btn = gr.Button("查詢條款", variant="primary")
        with gr.Column(scale=2):
            ans = gr.Textbox(label="📌 顧問回答", lines=10)
            src = gr.Textbox(label="📄 依據條款", lines=5)
    
    btn.click(ask_insurance, inputs=q, outputs=[ans, src])

# 啟動時自動開啟瀏覽器
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
