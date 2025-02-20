from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

region_name = 'eu-west-1'
foundation_model = 'anthropic.claude-3-sonnet-20240229-v1:0'
knowledge_base_id = 'WNCKQMM4ZA'
data_source_id = 'FAADRXS34P'

session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
    region_name=region_name
)
bedrock_client = session.client('bedrock-runtime', region_name=region_name)
bedrock_agent_runtime_client = session.client('bedrock-agent-runtime', region_name=region_name)

from collections import defaultdict
import os

@app.post("/retrieve/")
async def retrieve_documents(request: Request):
    body = await request.json()
    question = body.get('question', '')
    print(question)
    try:
        response = bedrock_agent_runtime_client.retrieve(
            retrievalQuery= {
                'text': question
            },
            knowledgeBaseId=knowledge_base_id,
            retrievalConfiguration= {
                'vectorSearchConfiguration': {
                    'numberOfResults': 5,
                    'overrideSearchType': "HYBRID", # optional
                }
            }
        )

        retrieval_results = response.get("retrievalResults", [])
        processed_results = []

        for doc in retrieval_results:
            text = doc.get("content", {}).get("text", "Text not found")
            summary = text_processor(
                "Summarize this document in 50 words or less. Do NOT add any extra words before or after it.", 
                text
            )
            relevant_text = f'...{text_processor(
                "Find the most relevant paragraph from the text below. Do NOT add any extra words before or after it.", 
                text
            )}...'
            processed_results.append({
                "summary": summary,
                "relevant_text": relevant_text,
                "response": doc
            })

            print (processed_results)
        return {"query": question, "results": processed_results}
    except Exception as e:
        print(e)
        return {"error": str(e)}

def text_processor(prompt, text):            
    response = bedrock_client.invoke_model(
        modelId=foundation_model,
        body=json.dumps({
            "max_tokens": 40,
            "system": "You are an AI assistant that processes text. You must STRICTLY follow the user's instructions without adding any introductory, explanatory, or concluding words. Your response must contain ONLY what the user explicitly requests—nothing more, nothing less.",
            "messages": [
                {"role": "user", "content": f"{prompt}\n\n{text}"}
            ],
            "anthropic_version": "bedrock-2023-05-31"
        })
    )

    response_body = json.loads(response["body"].read().decode("utf-8"))
    return response_body.get("content", [{}])[0].get("text", "")

# Run the API using:
# uvicorn huddleai:app --host 0.0.0.0 --port 8000 --reload