from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time

ollama = ChatOllama(model="llama3ko")

DB_PATH = "/home/eunjikim22/rag_project/vectorstores/db"
vectorstore = Chroma(persist_directory=DB_PATH,
                     embedding_function=OllamaEmbeddings(model='llama3ko'))
retriever = vectorstore.as_retriever()

# # 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "당신은 Q&A 작업의 보조자입니다."
    "검색된 컨텍스트의 다음 부분을 사용하여 질문에 대답하십시오. "
    "질문. 답을 모르면 이렇게 말하세요 "
    "모르겠습니다. 최대 3개의 문장을 사용하세요."
    "간결하게 대답하세요."
    "\n\n"
    "{context}"
)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(ollama, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# response = rag_chain.invoke({"input": "제주 다크투어리즘 관광지 2개만 골라서 추천해줘"})
# print(response["answer"])

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

chat_history = []
start = time.time()
question = "제주 다크 투어리즘 관광지 한가지 추천해줘"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)
print(ai_msg_1["answer"])

second_question = "그에 얽힌 역사적 사실도 알려줘"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])

end = time.time()
print(f'{end-start:.5f}sec')

'''
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

conversational_rag_chain.invoke(
    {"input": "제주 다크투어리즘 관광지 2개 추천해줘"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
'''

