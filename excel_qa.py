import re
import pandas as pd 
from custom_llm import CustomLLMModel
from custom_rag import CustomRAG
from langchain_community.document_loaders import UnstructuredExcelLoader


def load_excel_docs(files_list):
    docs = []
    for file in files_list:
        loader = UnstructuredExcelLoader(file)
        data = loader.load()
        docs.extend(data)
    return docs

# Remove extra spaces and blank lines
def clean_text(text):
    # Replace multiple spaces/newlines with a single space/newline
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with a single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'^\s+|\s+$', '', text)  # Remove leading/trailing whitespace
    text = re.sub(r'\b(?:[a-zA-Z] )+[a-zA-Z]\b',
                  lambda m: m.group(0).replace(' ', ''),
                  text)
    return text

if __name__ == "__main__":
    #load the excel files and clean the contents
    documents = load_excel_docs(["ProjectA.xlsx"])
    cleaned_docs = [clean_text(doc.page_content) for doc in documents]

    # Instantiate an LLM create a vector store to create the embeddings and store in Chroma vector DB
    local_llm = CustomLLMModel()
    vector_store = local_llm.create_vectorstore(input_text=cleaned_docs)

    #create the an instance of RAG class
    custom_rag = CustomRAG(model=local_llm)

    # Get the search query and respond to the questions
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = custom_rag.do_similarity_search(vector_store=vector_store, query=query)
        print(f"Response: {response}")