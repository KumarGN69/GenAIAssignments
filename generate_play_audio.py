from custom_llm import CustomLLMModel
from gtts import gTTS
import pygame

llm = CustomLLMModel()
client = llm.getclientinterface()

while True:
    query = input("Enter a theme for your story: (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    generated_content = client.generate(
        model=llm.MODEL_NAME,
        prompt= (
            f"Write a 3 paragraph story for children about the entered theme."
            f"Use age appropriate language and storyline using simple and plain english"
            f"Strictly output the story without any other comments"
            f"Theme:{query}"
        )
    )
    print(generated_content.response)
    # ----convert the generated text into audio------------
    tts = gTTS(text=generated_content.response, lang='en')
    tts.save("response.mp3")

    #---------- Play the audio---
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue