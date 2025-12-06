from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import simulation, training, hunting, knowledge, visualization

app = FastAPI(title="Leon Impala Simulation", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(hunting.router, prefix="/api/hunting", tags=["Hunting"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
app.include_router(visualization.router, prefix="/api/visualization", tags=["Visualization"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Leon Impala Simulation API"}
