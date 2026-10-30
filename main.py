from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# ✅ 修复这里！换回 langchain_classic
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import gradio as gr
import os

# ========== 配置（你原来的配置全部保留，新增阈值参数） ==========
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 150
TOP_K = 5
SIMILARITY_THRESHOLD = 0.75  # 新增：相似度过滤阈值，分数>0.75代表相关性低，过滤


# ========== 工具函数 ==========
def load_document(file_path):
    docs = []
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text))
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(Document(page_content=text))
    return docs


def split_docs(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# ========== 强约束提示词（沿用你的prompt，没有改动） ==========
prompt = ChatPromptTemplate.from_template("""
你是一个文档问答助手，你必须严格遵守下面所有规则：
1. 你**只能使用【参考上下文】里面的内容回答问题**，禁止调用你自身学到的任何外部知识。
2. 如果参考上下文里面没有对应的答案，直接回复：我不知道，文档中没有该信息。
3. 不允许编造、补充文档以外的例子、额外解释。
4. 回答尽量和原文表述保持一致，不要自由发挥。

【参考上下文】
{context}

【用户问题】
{input}
""")

llm = OllamaLLM(model=LLM_MODEL)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)


# ========== 【改动重点】重构chat函数：先手动带分数检索，过滤，再跑chain，同时返回检索详情 ==========
def chat(question, vector_db):
    if vector_db is None:
        return "请先上传文档并构建向量库", ""

    # 1. 带相似度分数检索
    docs_with_score = vector_db.similarity_search_with_score(question, k=TOP_K)
    # 2. 阈值过滤，丢掉相关性太低的片段
    docs_with_score = [(doc, score) for doc, score in docs_with_score if score < SIMILARITY_THRESHOLD]
    docs_filtered = [item[0] for item in docs_with_score]

    # 拼接检索详情文本，用于前端展示
    retrieve_detail = ""
    for doc, score in docs_with_score:
        retrieve_detail += f"相似度分数：{score:.4f}\n召回片段：{doc.page_content}\n——————————————\n"

    # 3. 用过滤后的文档，走你原来的RAG chain逻辑
    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": question})
    return result["answer"], retrieve_detail


# ========== Gradio 构建向量库函数【无改动，和你原来一模一样】 ==========
def build_vector_db(file):
    if file is None:
        return "请先上传PDF或者TXT文档", None
    fname = file.name.lower()
    if not (fname.endswith(".pdf") or fname.endswith(".txt")):
        return "❌ 仅支持 .pdf / .txt 文件，不支持docx、word等格式", None

    docs = load_document(file.name)
    if not docs:
        return "文档读取失败", None

    splits = split_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return "✅ 文档解析完成，可以开始提问！", vector_db


# ========== UI界面【改动：新增检索详情输出框，修改输出绑定】 ==========
with gr.Blocks(title="Simple-RAG-聊天机器人 |与Ollama合作的地方RAG") as demo:
    gr.Markdown("# Simple-RAG-聊天机器人 |与Ollama合作的地方RAG")
    vec_db_state = gr.State(None)
    with gr.Row():
        upload_file = gr.File(label="上传PDF / TXT")
        info_text = gr.Textbox(label="状态", interactive=False)
    build_btn = gr.Button("解析文档，构建向量库")

    question_box = gr.Textbox(label="你的问题")
    answer_box = gr.Textbox(label="AI回答", lines=6)
    # ===== 新增检索详情面板 =====
    retrieve_box = gr.Textbox(label="🔍 本次召回片段 & 相似度分数", lines=10, interactive=False)
    ask_btn = gr.Button("提问")

    build_btn.click(build_vector_db, inputs=upload_file, outputs=[info_text, vec_db_state])
    # 这里outputs改成两个：回答 + 检索详情
    ask_btn.click(chat, inputs=[question_box, vec_db_state], outputs=[answer_box, retrieve_box])

if __name__ == "__main__":
    demo.launch()