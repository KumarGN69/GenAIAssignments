from custom_llm import CustomLLMModel
import time
from langchain_chroma import Chroma
class CustomRAG:
    """
    Loads the website and reads the content
    Does several RAG operations including embedding into a vector DB, summarize, similarity search, generate graphs


    """
    # def __init__(self,website:str,search_str:str,prompt:str):
    def __init__(self, **kwargs):
        # self.model = LLMModel()
        self.model = kwargs.get('model')
        # self.search_str = kwargs.get("search_str")
        # self.prompt = kwargs.get("prompt")

    def get_summary(self,vector_store:Chroma):

        # create embedding for the documents, embedd into a vector store
        # vector_store = self.model.create_vectorstore(documents)

        # get the full content from the vector store for summarization
        doclist = vector_store.get()['documents']

        # get the Ollama Client interface to the model
        client = self.model.getclientinterface()

        # generate a llm response using client along with the RAG results
        generated_content = client.generate(
            model=self.model.MODEL_NAME,
            prompt=f"Summarize the provided context without missing key details. Context: {doclist}"
        )
        # print(type(generated_content.response))
        return generated_content.response

    def do_similarity_search(self, vector_store:Chroma, query:str):
        """instantiate the custom model and get the handle to it"""
        # model = LLMModel()

        # create a retriever from the vector store
        print('Creating a vector store retriever')

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 25}
        )

        # do a similarity search using the vector store retriever on specific search query

        doclist = retriever.invoke(query)
        # print(f'Retrieved documents : {doclist}')
        # get the Ollama Client interface to the model
        client = self.model.getclientinterface()

        # generate a llm response using client along with the RAG results
        print('Invoking the LLM for RAG based response')
        start = time.time()
        generated_content = client.generate(
            model=self.model.MODEL_NAME,
            prompt=f"Answer the user's query based on the provided context. Context: {doclist}"
        )
        end = time.time()
        print(f'Time taken to answer the query: {end-start} seconds')
        return generated_content.response
    