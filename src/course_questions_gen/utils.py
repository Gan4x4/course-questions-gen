from configparser import ConfigParser
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from course_questions_gen.prompts import load_prompts
import pandas as pd


@dataclass(frozen=True)
class GraphContext:
    model: str
    prompts_dir: str
    question_count: int
    output_path: str
    topics_path: str = "data/topics.csv"


def create_default_context() -> GraphContext:
    config = ConfigParser()
    config.read("settings.ini")

    return GraphContext(
        model=config["base"]["model"],
        prompts_dir=config["base"]["prompts_dir"],
        question_count=config.getint("extra", "question_count"),
        output_path=config["extra"]["output_path"],
        topics_path=config["extra"]["topics_path"],
    )


def create_llm(context: GraphContext):
    return ChatOpenAI(
        model=context.model,
        temperature=0,
    )


def create_prompts(context: GraphContext):
    return load_prompts(context.prompts_dir)


class TopicsCSV(object):

    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        self.section_col = 0

    def sections(self):
        x = self.df.iloc[:,self.section_col].dropna()
        return  self.to_list(x) 
    
    def to_list(self,x: pd.Series):
        return x[x.str.strip() != ""].unique().tolist() # remove empty lines    
    
    def topics(self, section: str):
        index = self.df.index[self.df.iloc[:, self.section_col] == section][0]
        topics_col = self.section_col +1
        index = index + 1
        topics = []
        while index < len(self.df):
            section_col_value = self.df.iloc[index, self.section_col]
            if not (pd.isna(section_col_value) or str(section_col_value).strip() == ""):
                break
            topics.append(str(self.df.iloc[index, topics_col]))
            index += 1
        return topics
