import whisper
from custom_llm import CustomLLMModel

model = whisper.load_model("turbo")
audio_file = "./response.mp3"
results = model.transcribe(audio_file)
transcript = results['text']

prompt = (
    f"Generate enough panels to create doodle style comic strip for children"
    f"with bubbles to indicate what characters are saying. using the storyline: {transcript}."
    f"Use your imagination but ensure language is simple and age appropriate "
    f"Output the panels in JSON format"
    f"Each panel should contain the details for the image and the text bubble"
    f"Strictly output the story without any other comments"
)

llm = CustomLLMModel()
client = llm.getclientinterface()

generated_content = client.generate(
    model = llm.MODEL_NAME,
    prompt = prompt
)

print(generated_content.response)