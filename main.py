from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_decisions, extract_questions
import os
from core.rag_engine import build_rag_chain, load_rag_chain, ask_question


load_dotenv()

def run_pipeline(source: str, language: str = "en") -> dict:
    """
    Run the entire pipeline: audio processing, transcription, summarization, and extraction.
    """

    print("starting ai assistant input...")
    chunks = process_input(source)
    print(f"Audio processing complete. Generated {len(chunks)} chunks.")

    transcript = transcribe_all(chunks, language=language)
    print("Transcription complete. Transcript length:", len(transcript))

    title = generate_title(transcript)
    print("Generated title:", title)

    summary = summarize(transcript)
    print("Generated summary:", summary)

    action_items = extract_action_items(transcript)
    print("Extracted action items:", action_items)  

    decisions = extract_decisions(transcript)
    print("Extracted decisions:", decisions)

    questions = extract_questions(transcript)
    print("Extracted questions:", questions)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "transcript": transcript,
        "rag_chain": rag_chain,

    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Enter language (english/hinglish): ").strip() or "english"
    results = run_pipeline(source, language=language)

    print("\n\n=== Final Results ===")
    print(f"Title: {results['title']}\n")
    print(f"Summary: {results['summary']}\n")
    print(f"Action Items:\n{results['action_items']}\n")
    print(f"Decisions:\n{results['decisions']}\n")
    print(f"Questions:\n{results['questions']}\n")

    # chat with your meeting via rag

    print("\nYou can now ask questions about the meeting. Type 'exit' to quit.")
    rag_chain = results["rag_chain"]
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye!")
            break
        if not question:
            print("Please enter a valid question.")
            continue
        answer = ask_question(rag_chain, question)
        print(f"Answer: {answer}")

        
            