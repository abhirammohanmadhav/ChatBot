from langchain.retrievers import ContextualCompressionRetriever, CohereRagRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
import os

def extract_file_data(root_path, folder_name) -> dict:
    documents = {}

    documents_folder = os.path.join(root_path, folder_name)

    if not os.path.exists(documents_folder):
        return documents
    
    files = os.listdir(documents_folder)
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(documents_folder, file), 'r', encoding='utf-8') as f:
                data = f.read()
            documents[file] = data
    return documents

if __name__ == "__main__":
    root_path = r"C:\D\NLP_Project\Own_creation\Customer_Support_chat_bot"
    folder_name = "Documents"
    all_docs = list(extract_file_data(root_path, folder_name).values()) 

    user_query = "How do I troubleshoot slow internet?"
    api_key = os.getenv("COHERE_API_KEY") 

    # Create cohere's chat model and embeddings objects
    cohere_chat_model = ChatCohere(cohere_api_key="{API-KEY}")
    cohere_embeddings = CohereEmbeddings(cohere_api_key="{API-KEY}")

    # Load text files and split into chunks, you can also use data gathered elsewhere in your application
    raw_documents = all_docs
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Create a vector store from the documents
    db = Chroma.from_documents(documents, cohere_embeddings)
    input_docs = db.as_retriever().get_relevant_documents(user_query)

    # Create the cohere rag retriever using the chat model 
    rag = CohereRagRetriever(llm=cohere_chat_model)
    docs = rag.get_relevant_documents(
        user_query,
        source_documents=input_docs,
    )

    # Print the documents
    for doc in docs[:-1]:
        print(doc.metadata)
        print("\n\n" + doc.page_content)
        print("\n\n" + "-" * 30 + "\n\n")

    # Print the final generation 
    answer = docs[-1].page_content
    print(answer)
    
    # Print the final citations 
    citations = docs[-1].metadata['citations']
    print(citations)