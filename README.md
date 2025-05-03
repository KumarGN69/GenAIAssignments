# GenAIAssignments
Question&Answering using RAG Embedding and LLMs
    custom_llm.py
        1. Custom local llm using Ollam and mistral model
    custom_rag.py
        1. Custom local RAG implementation using nomic_embed_text embedding model
    excel_qa.py
        1. This script is designed to generate a question and answer pair based on a given excel files.
        2. It uses the Ollama based local llm and RAG implementation to generate the question and answer.
    question_answer.py
        1. This script is designed to generate a question and answer pair based on a given pdf.
        2. It uses the Ollama based local llm and RAG implementation to generate the question
    code_generation.py
        1. To auto generate code using local LLM
        2. Tried with mistral, qwen2.5-coder and llama3.2. Llama3.2's out put was better structured
    multivariate_salary_prediction.py
        1. used data set from Kaggle.com
        2. Used multi-layer NeuralNetworks based regression model for predictions
        3. Model accuracy needs to be improved
        4. Saved the trained model and weights into json and h5 formats for reused
    loan_approval.py
        1. Used KNNClassifier
        2. Used data set from Kaggle.com
        3. Used GridSearchCV for hyperparameter tuning
        4. Model accuracy needs to be improved
    generate_play_audio.py
        1. Ollama based llama3.2 to generate text based on the input theme
        2. Used gTTS for text to speech conversion
        3. Used pygame for playing the audio
    audio_text_generate.py
        1. Used OpenAI-Whisper to convert audio to text
        2. Used Ollama with llama3.2 running locally to generate panels that can be used to generate a comic strips
    tree_of_thoughts.py
        1. Used mistral:latest and llama3.2 instead of gemini and openai using Ollama locally on my laptop
        2. Prompted the user to enter their thoughts to brainstorm
        3. Generated a markdown file with the ideas and expanded thoughts with reasoning
    find_word_embedding.py
        1. used python library to download a specific word embedding model
        2. Prompted the user to enter the word
        3. Searched the downloaded model for the entered word
        4. If available in corpus printed the embedding else printed that word is not found

