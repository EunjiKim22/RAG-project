from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

DATA_PATH = "/home/eunjikim22/rag_project/data"
DB_PATH = "/home/eunjikim22/rag_project/vectorstores/db1/"


def create_vector_db():
    # 부르고 pdf 갯수 반환
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Processed {len(documents)} pdf files")

    # 텍스트 분할 문서 단위로 자름
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print("split complete")

    # 인덱싱 분할된 문서를 검색가능한 형태로 만듬
    vectorstore = Chroma.from_documents(documents= texts, 
                                        embedding=OllamaEmbeddings(model="llama3ko"), persist_directory=DB_PATH)      

if __name__ == "__main__":
    create_vector_db()