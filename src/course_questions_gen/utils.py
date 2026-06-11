from configparser import ConfigParser
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from course_questions_gen.prompts import load_prompts
import pandas as pd


def read_settings() -> ConfigParser:
    config = ConfigParser()
    config.read("settings.ini")
    return config


@dataclass
class GraphContext:
    model: str
    prompts_dir: str
    question_count: int
    output_path: str
    topics_path: str
    llm_timeout_seconds: int
    llm_max_retries: int

    def __init__(
        self,
        model: str | None = None,
        prompts_dir: str | None = None,
        question_count: int | None = None,
        output_path: str | None = None,
        topics_path: str | None = None,
        llm_timeout_seconds: int | None = None,
        llm_max_retries: int | None = None,
    ):
        settings = read_settings()

        if model is not None:
            self.model = model
        else:
            self.model = settings["base"]["model"]

        if prompts_dir is not None:
            self.prompts_dir = prompts_dir
        else:
            self.prompts_dir = settings["base"]["prompts_dir"]

        if question_count is not None:
            self.question_count = question_count
        else:
            self.question_count = settings.getint("extra", "question_count")

        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = settings["extra"]["output_path"]

        if topics_path is not None:
            self.topics_path = topics_path
        else:
            self.topics_path = settings["extra"]["topics_path"]

        if llm_timeout_seconds is not None:
            self.llm_timeout_seconds = llm_timeout_seconds
        else:
            self.llm_timeout_seconds = settings.getint("extra", "llm_timeout_seconds")

        if llm_max_retries is not None:
            self.llm_max_retries = llm_max_retries
        else:
            self.llm_max_retries = settings.getint("extra", "llm_max_retries")


def create_default_context() -> GraphContext:
    return GraphContext()


def create_llm(context: GraphContext):
    return ChatOpenAI(
        model=context.model,
        temperature=0,
        timeout=context.llm_timeout_seconds,
        max_retries=context.llm_max_retries,
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
