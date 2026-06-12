from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.3
    )

def split_transcript(transcript: str) -> list[str]:
    """
    Split the transcript into smaller pieces for better summarization.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    """
    Summarize the transcript using Mistral.
    """

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert meeting summarizer. combine these partial summaries"
             " into one final professional meeting summary in bulllet points.",),
            ("human", "{text}"),
        ]
    )   

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript: str) -> str:
    """
    Generate a concise title for the meeting based on the transcript.
    """

    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert meeting summarizer. Generate a concise title for a meeting based on its transcript."),
                ("human", "{text}"),
            ]
        ) | llm | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])
