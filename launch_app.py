#!/usr/bin/env python3
"""
NCAA March Madness 2025 Predictor - Launch Script
Easy way to start the web application
"""

import uvicorn
import webbrowser
import time
import threading

def open_browser():
    """Open the web browser after a delay"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("🏀 Starting NCAA March Madness Predictor 2025...")
    print("📊 Loading ML models and historical data...")
    print("🌐 Web application will be available at: http://localhost:8000")
    print("📖 API documentation available at: http://localhost:8000/docs")
    print("\n" + "="*60)
    print("🎯 Features Available:")
    print("   • Dashboard with tournament statistics")
    print("   • Team vs Team predictions")
    print("   • Tournament bracket simulation") 
    print("   • Data analytics and insights")
    print("   • REST API for predictions")
    print("="*60 + "\n")
    
    # Start browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 