from configparser import ConfigParser
from langchain_openai import ChatOpenAI
from course_questions_gen.graph import GraphContext
from course_questions_gen.prompts import  load_prompts



def create_default_context() -> GraphContext:
    
    config = ConfigParser()
    config.read("settings.ini")
    
    llm =ChatOpenAI(
        model=config["base"]["model"],
        temperature=0,
    )

    prompts = load_prompts(config["base"]["prompts_dir"])
    
    return GraphContext(
        llm=llm,
        prompts=prompts,
    )

