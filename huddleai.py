from fastapi import FastAPI
import boto3
import json
import os
import re

app = FastAPI()
region_name = "eu-west-1"
foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
knowledge_base_id = "WNCKQMM4ZA"

session = boto3.Session(profile_name="AdministratorAccess-940482414003")  # Use your AWS SSO profile name
bedrock_client = session.client('bedrock-runtime', region_name=region_name)
bedrock_agent_runtime_client = session.client('bedrock-agent-runtime', region_name=region_name)

@app.get("/query/")
def query_knowledge_base(question: str):
    try:
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={
                "text": question
            },
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    'knowledgeBaseId': knowledge_base_id,
                    "modelArn": "arn:aws:bedrock:{}::foundation-model/{}".format(region_name, foundation_model),
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults":5
                        } 
                    }
                }
            }
        )

        response_body = response['output']['text']
        return {"query": question, "response": response_body}

    except Exception as e:
        return {"error": str(e)}

@app.get("/retrieve/")
def retrieve_documents(question: str, numberOfResults: int):
    try:
        response = bedrock_agent_runtime_client.retrieve(
            retrievalQuery= {
                'text': question
            },
            knowledgeBaseId=knowledge_base_id,
            retrievalConfiguration= {
                'vectorSearchConfiguration': {
                    'numberOfResults': numberOfResults,
                    'overrideSearchType': "HYBRID", # optional
                }
            }
        )

        retrieval_results = response.get("retrievalResults", [])

        def summarize_text(prompt, text):            
            # Correct request body format
            response = bedrock_client.invoke_model(
                modelId=foundation_model,
                body=json.dumps({
                    "max_tokens": 50,
                    "system": "You are an assistant that summarizes text. Provide a concise summary in 50 words or less.",
                    "messages": [
                        {"role": "user", "content": f"{prompt}\n\n{text}"}
                    ],
                    "anthropic_version": "bedrock-2023-05-31"
                })
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body.get("content", "Summary unavailable")

        processed_results = []
        for doc in retrieval_results:
            content = doc.get("content", "")
            summary = summarize_text("Summarize this document in 50 words or less.", content)
            relevant_text = summarize_text("Extract the relevant paragraph without modifying the extracted paragraph, and please help me to add '...' from the start and the end of the extracted paragraph.", content)

            processed_results.append({
                "summary": summary,
                "relevant_text": relevant_text,
                "response": doc
            })

        return {"query": question, "response": processed_results}
    except Exception as e:
        return {"error": str(e)}

# Run the API using:
# uvicorn huddleai:app --host 0.0.0.0 --port 8000 --reload


