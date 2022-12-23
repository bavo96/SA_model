# FastAPI
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import uvicorn

from typing import Union
import requests
import numpy as np

from model import SentimentAnalysis
        
sa_model = SentimentAnalysis("./saved_model_2")

description = """
API to do sentiment analysis.

"""

tags_metadata = [
    {
        "name": "predict",
        "description": """Get bot prediction for sentiment analysis. Output: positive, negative.
        """
            
    },
]

app = FastAPI(
        title='Sentiment Analysis Application',
        description=description,
        version='1',
        terms_of_service="http://example.com/terms/",
        contact={
            "name": "Vo Quoc Bang",
            "email": "bavo.imp@gmail.com",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        openapi_tags=tags_metadata)

class Request_Item(BaseModel):
    image_url: str = Field(default="", title="Human text that need to do sentiment analysis.")


class Response_Item(BaseModel):
    result: dict = Field(default={}, title="Results for n classes (, , ).")

@app.get('/')
def greeting():
    return "Sentiment Analysis Application."

@app.post('/predict',
        response_model=Response_Item,
        responses={
            200: {
                "description": "Sentiment Analysis result.",
                "content": {
                    "application/json": {
                        "example": {
                            "result" : {"positive": 1.0, "negative": 0.0}
                            }
                        }
                    }
                 }
            },

        ) 
def get_predict(
        item: Request_Item = Body(
            example={
                "text": "Sản phẩm rất tốt, phù hợp với giá tiền."
                }
            ),
         ):
    text = item.dict().get('text', None)

    result = sa_model.inference(text)
    print(result)
    return {"result": result}

@app.get('/healthcheck')
def healthcheck():
    return "API is alive."

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


