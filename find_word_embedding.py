import gensim.downloader as api 

model = api.load("word2vec-google-news-300")
while True:
    user_input = input(f"Enter your word to get the embedding :")
    if user_input.lower() == "exit":
        break
    if user_input in model:
        vector = model[user_input]
        print(f"Embedding for the {user_input} is {vector} \n")
    else:
        print(f"Embedding for {user_input} not found in the corpus \n")    