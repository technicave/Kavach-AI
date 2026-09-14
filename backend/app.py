from fastapi import FastAPI

app = FastAPI(
    title="Kavach-AI",
    description="AI-Based Fake Identity & Document Screening System",
    version="1.0.0"
)


@app.get("/api/v1")
async def starting():
    return {
        "message": "Welcome to the Kavach-AI"
    }


@app.post("/api/v1/passport")
async def passport():
    return {
        "document_type": "passport",
        "status": "received"
    }


@app.post("/api/v1/aadhaar")
async def aadhaar():
    return {
        "document_type": "aadhaar",
        "status": "received"
    }
