import pypdf, re
from custom_llm import CustomLLMModel
from custom_rag import CustomRAG

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

    # Load the PDF file
    reader = pypdf.PdfReader('./Inputs/EU_AI_ACT.pdf')
    #extract the contents
    docs = [page.extract_text() for page in reader.pages]
    #clean the contents
    cleaned_docs = [clean_text(doc) for doc in docs]

    # create a vector store which store
    local_llm = CustomLLMModel()

    vector_store = local_llm.create_vectorstore(input_text=cleaned_docs)
    custom_rag = CustomRAG(model=local_llm)

    # Get the search query and respond to the questions
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        # query_embedding = local_llm.create_embedding().embed_query(query)
        response = custom_rag.do_similarity_search(vector_store=vector_store, query=query)
        print(f"Response: {response}")