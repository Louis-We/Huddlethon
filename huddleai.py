from fastapi import FastAPI
import boto3
import json
import os
import re

app = FastAPI()
region_name = 'eu-west-1'
foundation_model = 'anthropic.claude-3-sonnet-20240229-v1:0'
knowledge_base_id = 'WNCKQMM4ZA'
data_source_id = 'FAADRXS34P'

session = boto3.Session(profile_name="AdministratorAccess-940482414003")  # Use your AWS SSO profile name
bedrock_client = session.client('bedrock-runtime', region_name=region_name)
bedrock_agent_runtime_client = session.client('bedrock-agent-runtime', region_name=region_name)

from collections import defaultdict
import os

@app.get("/retrieve/")
def retrieve_documents(question: str, numberOfResults: int):
    try:
        response = bedrock_agent_runtime_client.retrieve(
            retrievalQuery={'text': question},
            knowledgeBaseId=knowledge_base_id,
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': numberOfResults,
                    'overrideSearchType': "SEMANTIC",
                }
            }
        )

        retrieval_results = response.get("retrievalResults", [])
        grouped_results = defaultdict(list)

        # Group chunks by file name
        for doc in retrieval_results:
            file_name = os.path.basename(doc.get('location', {}).get('s3Location', {}).get('uri', 'Unknown'))
            text = doc.get("content", {}).get("text", "Text not found")
            grouped_results[file_name].append(text)

        # Process grouped files
        processed_results = []
        for file_name, chunks in grouped_results.items():
            summary = text_processor(
                f"Based on the following chunks, explain why this file: {file_name} is relevant to this question: {question}. Answer it in 50 words or less.",
                " ".join(chunks)
            )
            relevant_texts = []
            for chunk in chunks[:2]:  # Pick at least 2 chunks per file
                relevant_text = text_processor(
                    f"Find the paragraph from the text below that is most relevant to this question: {question}. Do NOT add any extra words before or after it.",
                    chunk
                )
                relevant_texts.append(f"...{relevant_text}...")

            processed_results.append({
                "file_name": file_name,
                "summary": summary,
                "relevant_texts": relevant_texts
            })

        return {"query": question, "response": processed_results}
    except Exception as e:
        return {"error": str(e)}

def text_processor(prompt, text):            
    response = bedrock_client.invoke_model(
        modelId=foundation_model,
        body=json.dumps({
            "max_tokens": 40,
            "system": f"You are an AI assistant that processes text. You must STRICTLY follow the user's instructions without adding any introductory, explanatory, or concluding words. Your response must contain ONLY what the user explicitly requests—nothing more, nothing less.",
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