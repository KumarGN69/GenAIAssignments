from custom_llm import CustomLLMModel
import json
from pydantic import BaseModel
from typing import List

# define the structure for parsing the llm output
class Idea(BaseModel):
    name: str
    description:str

class IdeasList(BaseModel):
    solutions: List[Idea]

# instantiate the local llm
llm = CustomLLMModel()

# prompt the user for their idea to brainstorm
thought = input(f"Enter your idea for brainstorming :" )

client = llm.getclientinterface()
ideas = client.generate(
    model = "mistral:latest",
    prompt = (
        f"Brainstorm a solution path for : {thought}"
        f"Respond in the following JSON format"
        """
        {
            "solutions" :[
                {"name":"", "description":""},
                {"name":"", "description":""},
                {"name":"", "description":""},
            ]
        }
        """
    
    ),
)
#parse the llm output into a structured list using pydantic classes
ideas_list = IdeasList.model_validate_json(ideas.response)

#Brain storm and write the ideas to a markdown file
with open("TreeOfThoughts.md","w",encoding="utf-8") as f:
    f.writelines(f"# Topic: {thought}"+"\n")
    for idea in ideas_list.solutions:       
        expanded_thought = client.generate(
            model = "llama3.2",
            prompt = (f"Expand and detail the reasoning step by step for \n: {idea.description}")
        )
        print(f"{idea.name}: {expanded_thought.response} \n")
        f.writelines(f"- ## Idea: {idea.name} :" +"\n" )
        f.writelines(f"-- ### Description: {idea.description} :"+"\n" )        
        f.writelines(f"--- #### {expanded_thought.response}"+"\n")
