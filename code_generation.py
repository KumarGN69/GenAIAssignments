from custom_llm import CustomLLMModel

prompt = (  
            f"Act as a Python developer to do the following:" 
            f"1. Ask the user to provide the name of the csv file"
            f"2. Write code that reads the records.csv file and print duplicate records" 
            f"3. Removes the duplicate rows " 
            f"4. Saves the updated records to a new file named updated_records.csv"
            f"5. Output ONLY raw Python code — do NOT include markdown formatting, code fences at the beginning or end"
        )

model = CustomLLMModel()

client = model.getclientinterface()
generated_content = client.generate(
        model= model.MODEL_NAME,
        prompt = prompt
    )
with open("generated_code.py","w") as file:
    file.write(generated_content.response)
    file.close()