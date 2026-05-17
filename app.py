from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run_deep_research

app = FastAPI(
    title="Legal Compliance Agent API",
    description="API for evaluating legality of AI data collection workflows",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ComplianceRequest(BaseModel):
    platform_activity: str


# Serve frontend
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# Response endpoint
@app.post("/check-compliance")
def check_compliance(request: ComplianceRequest):
    try:
        result = run_deep_research(request.platform_activity)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
