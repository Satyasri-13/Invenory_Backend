from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import upload, inventory, risk, alerts, correlation, model, rootcause

app = FastAPI(title="Inventory Sense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(inventory.router)
app.include_router(risk.router)
app.include_router(alerts.router)
app.include_router(correlation.router)
app.include_router(model.router)
app.include_router(rootcause.router)
@app.get("/health")
def health():
    return {"status": "Backend running"}
