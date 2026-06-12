import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from core.vector_store import load_vector_store, get_retriever, build_vector_store


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.3
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant that answers questions about a meeting based on the meeting transcript."),
            ("human", "Answer the question based on the following relevant excerpts from the transcript: {context}\n\nQuestion: {question}"),
        ]
    )

    # Fulll LCEL rag pipeline

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain

#load rag chain function and vector store in memory for faster response to user queries.
def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever()

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant that answers questions about a meeting based on the meeting transcript."),
            ("human", "Answer the question based on the following relevant excerpts from the transcript: {context}\n\nQuestion: {question}"),
        ]
    )
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    """
    Ask a question to the RAG chain and get an answer.
    """
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer