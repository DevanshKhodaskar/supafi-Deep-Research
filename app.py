from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import run_deep_research

app = FastAPI(
    title="Legal Compliance Agent API",
    description="API for evaluating legality of AI data collection workflows",
    version="1.0"
)

# Request schema
class ComplianceRequest(BaseModel):
    platform_activity: str


# Response endpoint
@app.post("/check-compliance")
def check_compliance(request: ComplianceRequest):
    try:
        result = run_deep_research(request.platform_activity)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/")
def root():
    return {"message": "Legal Compliance Agent API running"}