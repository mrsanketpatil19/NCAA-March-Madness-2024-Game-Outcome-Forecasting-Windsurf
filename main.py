#!/usr/bin/env python3
"""
NCAA March Madness 2025 Prediction Web Application
A professional FastAPI app for tournament predictions with sporty UI
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import os
import json
from typing import Optional, List, Dict
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
from models.predictor import NCAAPredictor

# Initialize FastAPI app
app = FastAPI(
    title="🏀 NCAA March Madness Predictor 2025",
    description="Advanced ML-powered tournament prediction system",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the ML predictor
predictor = NCAAPredictor()

# Initialize predictor on startup
@app.on_event("startup")
async def startup_event():
    """Load ML model and data on startup"""
    try:
        await predictor.load_model()
        print("🏀 NCAA Predictor loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    try:
        # Get basic stats for dashboard
        stats = await predictor.get_dashboard_stats()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "stats": stats,
            "title": "NCAA March Madness 2025 Predictor"
        })
    except Exception as e:
        print(f"Error in home route: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "stats": {},
            "title": "NCAA March Madness 2025 Predictor",
            "error": str(e)
        })

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Team prediction page"""
    try:
        teams = await predictor.get_teams_list()
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "teams": teams,
            "title": "Team vs Team Prediction"
        })
    except Exception as e:
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "teams": [],
            "title": "Team vs Team Prediction",
            "error": str(e)
        })

@app.get("/bracket", response_class=HTMLResponse)
async def bracket_page(request: Request):
    """Tournament bracket simulation page"""
    try:
        champions = await predictor.get_predicted_champions()
        return templates.TemplateResponse("bracket.html", {
            "request": request,
            "champions": champions,
            "title": "Tournament Bracket Simulation"
        })
    except Exception as e:
        return templates.TemplateResponse("bracket.html", {
            "request": request,
            "champions": {},
            "title": "Tournament Bracket Simulation",
            "error": str(e)
        })

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Data analytics and insights page"""
    try:
        # Generate analytics charts
        charts = await predictor.generate_analytics_charts()
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "charts": charts,
            "title": "Tournament Analytics & Insights"
        })
    except Exception as e:
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "charts": {},
            "title": "Tournament Analytics & Insights",
            "error": str(e)
        })

# API Endpoints
@app.post("/api/predict-matchup")
async def predict_matchup(
    team1_id: int = Form(...),
    team2_id: int = Form(...)
):
    """API endpoint to predict outcome of a specific matchup"""
    try:
        result = await predictor.predict_game(team1_id, team2_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/teams")
async def get_teams():
    """Get list of all teams"""
    try:
        teams = await predictor.get_teams_list()
        return JSONResponse(content=teams)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/team-stats/{team_id}")
async def get_team_stats(team_id: int):
    """Get detailed stats for a specific team"""
    try:
        stats = await predictor.get_team_stats(team_id)
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

@app.get("/api/tournament-predictions")
async def get_tournament_predictions():
    """Get full tournament predictions"""
    try:
        predictions = await predictor.get_full_tournament_predictions()
        return JSONResponse(content=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/upset-predictions")
async def get_upset_predictions():
    """Get likely upset predictions"""
    try:
        upsets = await predictor.get_upset_predictions()
        return JSONResponse(content=upsets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/seed-performance")
async def get_seed_performance():
    """Get historical seed performance data"""
    try:
        data = await predictor.get_seed_performance_data()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/conference-strength")
async def get_conference_strength():
    """Get conference strength analysis"""
    try:
        data = await predictor.get_conference_strength_data()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 