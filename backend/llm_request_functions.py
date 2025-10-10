from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)



#function to clean chatGPT return
def clean_json_output(raw_output: str) -> str:
    if not raw_output:
        return "{}"

    # Strip whitespace
    raw_output = raw_output.strip()

    # Remove Markdown code block fences if present
    if raw_output.startswith("```"):
        # Remove opening ```json or ```
        raw_output = re.sub(r"^```[a-zA-Z]*\n?", "", raw_output)
        # Remove trailing ```
        raw_output = raw_output.rstrip("`").strip()

    return raw_output

#Function that makes API request to chatGPT model
def model_request(genes_batch):
    prompt = f"""
    For each of the following genes: {genes_batch}.
    Return a <20 word description of each. 
    Include with what disease they are associated with, organelles and pathways.
    Respond in valid JSON like: {{"GENE": "description", ...}}
    Make sure the json string doesn't have json keyword inside of it.
    Make sure each gene gets some sort of correct biological description and make sure each string is enclosed with "
    """

    response = client.responses.create(
            model="gpt-5-chat-latest",
            input=prompt,
            max_output_tokens=1000000,
            
    )

    result = clean_json_output(response.output_text)
    
    return json.loads(result)

#summarise backend error for the user
def summarize_traceback(tb):
    """Ask OpenAI synchronously to summarize a traceback in simple, user-friendly language."""
    prompt = f"""
    You are a helpful assistant summarizing backend errors for users.
    Here is the Python traceback:

    {tb}

    Write a concise, friendly explanation (1–2 sentences)
    describing what likely went wrong. If the issue is related to user input 
    then let the user know.
    If the issue is related to something being wrong with the code or server, and is not related to the user input,
    let them know that.
    Avoid technical jargon, be polite, and helpful.
    """

    try:
        response = client.responses.create(
            model="gpt-5-chat-latest",
            input=prompt,
            max_output_tokens=1000,
            
        )

        summary = response.output[0].content[0].text.strip()
    except Exception as e:
        summary = "An unexpected error occurred. Please check your input or try again."

    return summary