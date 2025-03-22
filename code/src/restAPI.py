from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EmailData(BaseModel):
    subject: str
    body: str

@app.post("/check-email")
def classify_email(email: EmailData):
    return {"message": f"Classifying email: {email.subject}"}