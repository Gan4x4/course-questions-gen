from configparser import ConfigParser
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from course_questions_gen.prompts import Prompts, load_prompts
import pandas as pd


@dataclass(frozen=True)
class GraphContext:
    llm: BaseChatModel
    prompts: Prompts
    question_count: int
    output_path: str


def create_default_context() -> GraphContext:
    config = ConfigParser()
    config.read("settings.ini")

    llm = ChatOpenAI(
        model=config["base"]["model"],
        temperature=0,
    )

    prompts = load_prompts(config["base"]["prompts_dir"])

    return GraphContext(
        llm=llm,
        prompts=prompts,
        question_count=config.getint("extra", "question_count"),
        output_path=config["extra"]["output_path"],
    )


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

