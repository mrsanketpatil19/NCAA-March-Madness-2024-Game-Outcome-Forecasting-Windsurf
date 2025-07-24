#!/usr/bin/env python3
"""
NCAA March Madness 2025 Prediction System - Simplified Demo
This demonstrates the core concepts from the full notebook.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb

def load_and_process_data():
    """Load and process NCAA tournament data"""
    print("🏀 Loading NCAA Tournament Data...")
    
    base_path = "./march-machine-learning-mania-2025/"
    
    # Load key datasets
    tourney_results = pd.read_csv(os.path.join(base_path, "MNCAATourneyDetailedResults.csv"))
    tourney_seeds = pd.read_csv(os.path.join(base_path, "MNCAATourneySeeds.csv"))
    teams = pd.read_csv(os.path.join(base_path, "MTeams.csv"))
    
    print(f"📊 Loaded {len(tourney_results)} tournament games from {tourney_results['Season'].min()}-{tourney_results['Season'].max()}")
    
    # Merge seed information
    tourney_results = tourney_results.merge(
        tourney_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how="left"
    ).rename(columns={'Seed': 'WSeed'}).drop(columns=['TeamID'])
    
    tourney_results = tourney_results.merge(
        tourney_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how="left"
    ).rename(columns={'Seed': 'LSeed'}).drop(columns=['TeamID'])
    
    # Merge team names
    tourney_results = tourney_results.merge(
        teams[['TeamID', 'TeamName']], left_on="WTeamID", right_on="TeamID", how="left"
    ).rename(columns={'TeamName': 'WTeamName'}).drop(columns=['TeamID'])
    
    tourney_results = tourney_results.merge(
        teams[['TeamID', 'TeamName']], left_on="LTeamID", right_on="TeamID", how="left"
    ).rename(columns={'TeamName': 'LTeamName'}).drop(columns=['TeamID'])
    
    return tourney_results, teams

def create_features(data):
    """Create features for machine learning"""
    print("⚙️ Creating Features...")
    
    # Extract seed numbers (remove region letters and a/b suffixes)
    data["WSeedNum"] = data["WSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
    data["LSeedNum"] = data["LSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
    
    # Create key features
    data["WinMargin"] = data["WScore"] - data["LScore"]
    data["WFG_Pct"] = data["WFGM"] / data["WFGA"]
    data["LFG_Pct"] = data["LFGM"] / data["LFGA"]
    data["W3P_Pct"] = data["WFGM3"] / data["WFGA3"] 
    data["L3P_Pct"] = data["LFGM3"] / data["LFGA3"]
    data["WFT_Pct"] = data["WFTM"] / data["WFTA"]
    data["LFT_Pct"] = data["LFTM"] / data["LFTA"]
    data["Seed_Diff"] = data["WSeedNum"] - data["LSeedNum"]
    
    # Fill any missing percentage values with 0
    for col in ["WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]:
        data[col] = data[col].fillna(0)
    
    return data

def prepare_ml_data(data):
    """Prepare data for machine learning by creating balanced dataset"""
    print("🔄 Preparing Machine Learning Dataset...")
    
    # Create winning team data (label = 1)
    winners = data.copy()
    winners["Win"] = 1
    
    # Create losing team data (label = 0) by swapping teams
    losers = data.copy()
    losers["Win"] = 0
    losers["Seed_Diff"] = -losers["Seed_Diff"]  # Flip perspective
    
    # Swap winner/loser columns
    swap_cols = {
        "WTeamID": "LTeamID", "LTeamID": "WTeamID",
        "WSeedNum": "LSeedNum", "LSeedNum": "WSeedNum", 
        "WFG_Pct": "LFG_Pct", "LFG_Pct": "WFG_Pct",
        "W3P_Pct": "L3P_Pct", "L3P_Pct": "W3P_Pct",
        "WFT_Pct": "LFT_Pct", "LFT_Pct": "WFT_Pct"
    }
    losers = losers.rename(columns=swap_cols)
    
    # Combine datasets
    final_data = pd.concat([winners, losers], ignore_index=True)
    
    # Define features and target
    features = ["Seed_Diff", "WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]
    X = final_data[features]
    y = final_data["Win"]
    
    print(f"📈 Created balanced dataset: {len(X)} samples with {len(features)} features")
    
    return X, y, final_data

def train_model(X, y):
    """Train XGBoost model"""
    print("🤖 Training XGBoost Model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train model
    params = {
        "objective": "binary:logistic",
        "learning_rate": 0.05,
        "max_depth": 4,
        "eval_metric": "logloss"
    }
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Evaluate
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Model Performance:")
    print(f"   📊 Accuracy: {accuracy:.3f}")
    print(f"   📈 AUC Score: {auc:.3f}")
    
    return model

def predict_champions(model, teams_data):
    """Predict 2025 champions using simplified logic"""
    print("🏆 Predicting 2025 Champions...")
    
    # Get recent high-performing teams (simplified)
    recent_data = teams_data[teams_data["Season"] >= 2020]
    
    # Calculate team performance metrics
    team_stats = recent_data.groupby(["WTeamID", "WTeamName"]).agg({
        "WSeedNum": "mean",
        "WFG_Pct": "mean", 
        "W3P_Pct": "mean",
        "WFT_Pct": "mean",
        "WinMargin": "mean"
    }).reset_index()
    
    # Sort by performance (lower seed number = better, higher shooting = better)
    team_stats["Overall_Score"] = (
        (17 - team_stats["WSeedNum"]) * 0.3 +  # Seed ranking (inverted)
        team_stats["WFG_Pct"] * 0.25 +        # Field goal %
        team_stats["W3P_Pct"] * 0.2 +         # 3-point %
        team_stats["WFT_Pct"] * 0.15 +        # Free throw %
        team_stats["WinMargin"] * 0.1         # Win margin
    )
    
    top_teams = team_stats.nlargest(10, "Overall_Score")
    
    print("🎯 Top 10 Championship Contenders:")
    for i, (_, team) in enumerate(top_teams.iterrows(), 1):
        print(f"   {i:2d}. {team['WTeamName']} (Avg Seed: {team['WSeedNum']:.1f}, Score: {team['Overall_Score']:.3f})")
    
    predicted_champion = top_teams.iloc[0]["WTeamName"]
    print(f"\n🏆 PREDICTED 2025 NCAA CHAMPION: {predicted_champion}")
    
    return predicted_champion, top_teams

def main():
    """Main execution function"""
    print("=" * 60)
    print("🏀 NCAA MARCH MADNESS 2025 PREDICTION SYSTEM")
    print("=" * 60)
    
    try:
        # Load and process data
        data, teams = load_and_process_data()
        
        # Create features
        data = create_features(data)
        
        # Prepare ML data
        X, y, ml_data = prepare_ml_data(data)
        
        # Train model
        model = train_model(X, y)
        
        # Predict champions
        champion, top_teams = predict_champions(model, data)
        
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 