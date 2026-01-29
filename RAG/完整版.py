import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ======================
# 3. 載入 Qwen2-1.5B (針對 6GB VRAM 優化)
# ======================
model_id = "Qwen/Qwen2-1.5B-Instruct" 

print(f"🔥 正在以原生 FP16 模式載入 {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)

# 【關鍵點 1】在 pipeline 中加入停止與懲罰參數
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,    # 降低隨機性，讓回答更準確
    top_p=0.9,
    repetition_penalty=1.1, # 抑制模型「鬼打牆」重複說話
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# ======================
# 4. Prompt 與 Chain
# ======================
# 【關鍵點 2】改用 Qwen 專用的 ChatML 格式標籤
template = """<|im_start|>system
你是一位專業的保險顧問，請根據提供的條款內容，精確且簡潔地回答問題。如果條款中沒提到，請回答不知道。回答完畢請停止。
<|im_end|>
<|im_start|>user
條款內容：
{context}

問題：{question}
<|im_end|>
<|im_start|>assistant
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

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
        result = qa.invoke({"query": question})
        raw_answer = result["result"]
        
        # 【關鍵點 3】清理 LLM 輸出的標籤內容
        # 由於 LLM 可能會回傳包含 Prompt 的全文，我們只取 assistant 之後的內容
        if "<|im_start|>assistant" in raw_answer:
            answer = raw_answer.split("<|im_start|>assistant")[-1].strip()
        else:
            answer = raw_answer.strip()
            
        # 移除可能殘留的結束符號
        answer = answer.replace("<|im_end|>", "").strip()
            
        sources = "\n\n".join([f"📄 條款摘錄：\n{doc.page_content[:300]}" for doc in result["source_documents"]])
        return answer, sources
    except Exception as e:
        return f"發生錯誤：{str(e)}", ""

# ======================
# 6. 啟動介面
# ======================
with gr.Blocks(title="保險 RAG 系統") as demo:
    gr.Markdown("# 🧳 輕量版保險條款問答系統\n針對 6GB 顯存優化 (Qwen2-1.5B) - 已修復重複問題")
    with gr.Row():
        with gr.Column(scale=1):
            q = gr.Textbox(label="請輸入您的問題", placeholder="例如：行李遺失後該如何申請理賠？")
            btn = gr.Button("查詢條款", variant="primary")
        with gr.Column(scale=2):
            ans = gr.Textbox(label="📌 顧問回答", lines=10)
            src = gr.Textbox(label="📄 依據條款", lines=5)
    
    btn.click(ask_insurance, inputs=q, outputs=[ans, src])

# 啟動時自動開啟瀏覽器
if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
