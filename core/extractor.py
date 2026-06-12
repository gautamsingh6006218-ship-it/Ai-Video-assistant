# Actionable items, decision, and question extraction from video transcripts.   

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2
    )

def build_chain(system_prompt: str):

    llm = get_llm()

    return (RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()        
    ) 

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an assistant that extracts actionable items from meeting transcripts. "
        "Identify and list all actionable items in the following transcript. "
        "An actionable item is a task that someone needs to do, often with a deadline or specific instructions. "
        "Format the output as a bullet-point list of actionable items."
    )

    return chain.invoke(transcript)


def extract_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an assistant that extracts decisions from meeting transcripts. "
        "Identify and list all decisions made in the following transcript. "
        "A decision is a conclusion or resolution reached after discussion. "
        "Format the output as a bullet-point list of decisions."
    )

    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an assistant that extracts questions from meeting transcripts. "
        "Identify and list all questions asked in the following transcript. "
        "Format the output as a bullet-point list of questions."
    )

    return chain.invoke(transcript)

