from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage

import gradio as gr

ollama = ChatOllama(model="llama3ko")

DB_PATH = "/home/eunjikim22/rag_project/vectorstores/db"
vectorstore = Chroma(persist_directory=DB_PATH,
                     embedding_function=OllamaEmbeddings(model='llama3ko'))
retriever = vectorstore.as_retriever()

system_prompt = (
    "당신은 Q&A 작업의 보조자입니다."
    "검색된 컨텍스트의 다음 부분을 사용하여 질문에 대답하십시오. "
    "질문. 답을 모르면 이렇게 말하세요 "
    "모르겠습니다. 최대 3개의 문장을 사용하세요."
    "간결하게 대답하세요."
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "채팅 기록 및 최근 사용자 질문 제공"
    "채팅 기록의 맥락을 참조할 수 있습니다."
    "이해할 수 있는 독립형 질문을 공식화하세요."
    "채팅 기록이 없습니다. 질문에 대답하지 마세요."
    "필요하다면 다시 구성하고 그렇지 않으면 그대로 반환하세요."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    ollama, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(ollama, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# app start
def response(message, history):
     chat_history = []
     for human,ai in history:
         chat_history.extend(
              [HumanMessage(content=human),
               AIMessage(content=ai)])
     ollama_response=rag_chain.invoke({"input": message, "chat_history": chat_history})
     return ollama_response["answer"]

theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Jua"),"sans-serif"])

# primary_hue=gr.themes.colors.teal, 테마에서 시선을 끄는 색상
# secondary_hue=gr.themes.colors.cyan) 보조요소
# nautral_hue 텍스트 및 기타 중립 요소

gr.ChatInterface(
        fn=response,
        textbox=gr.Textbox(placeholder="질문을 입력해주세요", container=False, scale=7),
        # 채팅창의 크기를 조절한다.
        chatbot=gr.Chatbot(height=600),
        title="✨제주 역사 가이드 llm✨",
        # description="제주 역사 가이드 챗봇입니다.",
        theme=theme,
        examples=[["제주 다크 투어리즘 관광지 하나 알려줘"],["그에 얽힌 역사적 사실도 알려줘"], ["제주 역사에 대해 알려줘"]],
        retry_btn="다시보내기 ↩",
        undo_btn="이전챗 삭제 ❌",
        clear_btn="전챗 삭제 💫"
).launch()