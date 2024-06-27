from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import json

app = FastAPI()

sagemaker_runtime = boto3.client('sagemaker-runtime')

# Define a request model
class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    endpoint_name = "bits-deploy-new-staging"
    
    # Convert request data to JSON
    payload = json.dumps(request.data)

    try:
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload
        )

        # Parse the response
        result = json.loads(response['Body'].read().decode())
        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

