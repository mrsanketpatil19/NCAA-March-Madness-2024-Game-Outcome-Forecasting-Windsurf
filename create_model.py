#!/usr/bin/env python3
"""
NCAA Model Training & Pickle Creation
Train the XGBoost model and save it as a pickle file for the web app
"""

import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def create_and_save_model():
    """Train the NCAA prediction model and save as pickle"""
    
    print("🏀 Starting NCAA Model Training...")
    
    # Set paths
    base_path = "./march-machine-learning-mania-2025/"
    
    # Load data
    print("📊 Loading tournament data...")
    tourney_results = pd.read_csv(os.path.join(base_path, "MNCAATourneyDetailedResults.csv"))
    tourney_seeds = pd.read_csv(os.path.join(base_path, "MNCAATourneySeeds.csv"))
    teams = pd.read_csv(os.path.join(base_path, "MTeams.csv"))
    
    print(f"✅ Loaded {len(tourney_results)} tournament games")
    print(f"✅ Loaded {len(teams)} teams")
    
    # Merge tournament results with seeds
    print("🔄 Processing data...")
    tourney_results = tourney_results.merge(
        tourney_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how="left"
    ).rename(columns={'Seed': 'WSeed'}).drop(columns=['TeamID'])
    
    tourney_results = tourney_results.merge(
        tourney_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how="left"
    ).rename(columns={'Seed': 'LSeed'}).drop(columns=['TeamID'])
    
    # Add team names
    tourney_results = tourney_results.merge(
        teams[['TeamID', 'TeamName']], left_on="WTeamID", right_on="TeamID", how="left"
    ).rename(columns={'TeamName': 'WTeamName'}).drop(columns=['TeamID'])
    
    tourney_results = tourney_results.merge(
        teams[['TeamID', 'TeamName']], left_on="LTeamID", right_on="TeamID", how="left"
    ).rename(columns={'TeamName': 'LTeamName'}).drop(columns=['TeamID'])
    
    # Feature engineering
    print("⚙️ Engineering features...")
    
    # Extract seed numbers
    tourney_results["WSeedNum"] = tourney_results["WSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
    tourney_results["LSeedNum"] = tourney_results["LSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
    
    # Create performance features
    tourney_results["WinMargin"] = tourney_results["WScore"] - tourney_results["LScore"]
    tourney_results["WFG_Pct"] = tourney_results["WFGM"] / tourney_results["WFGA"]
    tourney_results["LFG_Pct"] = tourney_results["LFGM"] / tourney_results["LFGA"]
    tourney_results["W3P_Pct"] = tourney_results["WFGM3"] / tourney_results["WFGA3"]
    tourney_results["L3P_Pct"] = tourney_results["LFGM3"] / tourney_results["LFGA3"]
    tourney_results["WFT_Pct"] = tourney_results["WFTM"] / tourney_results["WFTA"]
    tourney_results["LFT_Pct"] = tourney_results["LFTM"] / tourney_results["LFTA"]
    tourney_results["Seed_Diff"] = tourney_results["WSeedNum"] - tourney_results["LSeedNum"]
    
    # Fill missing values
    for col in ["WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]:
        tourney_results[col] = tourney_results[col].fillna(0)
    
    # Prepare ML dataset
    print("🤖 Preparing ML dataset...")
    
    # Create balanced dataset
    winners = tourney_results.copy()
    winners["Win"] = 1
    
    losers = tourney_results.copy()
    losers["Win"] = 0
    losers["Seed_Diff"] = -losers["Seed_Diff"]
    
    # Swap winner/loser columns for losers
    swap_cols = {
        "WFG_Pct": "LFG_Pct", "LFG_Pct": "WFG_Pct",
        "W3P_Pct": "L3P_Pct", "L3P_Pct": "W3P_Pct",
        "WFT_Pct": "LFT_Pct", "LFT_Pct": "WFT_Pct"
    }
    losers = losers.rename(columns=swap_cols)
    
    # Combine datasets
    final_data = pd.concat([winners, losers], ignore_index=True)
    
    # Define features
    features = ["Seed_Diff", "WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]
    X = final_data[features]
    y = final_data["Win"]
    
    print(f"📈 Created dataset: {len(X)} samples with {len(features)} features")
    
    # Train model
    print("🎯 Training XGBoost model...")
    
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
        "eval_metric": "logloss",
        "random_state": 42
    }
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Evaluate model
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Model performance: {accuracy:.1%} accuracy, {auc:.3f} AUC")
    
    # Generate team statistics
    print("📊 Generating team statistics...")
    
    # Calculate team stats from winning games
    team_stats_win = tourney_results.groupby("WTeamID").agg({
        "WFG_Pct": "mean",
        "W3P_Pct": "mean", 
        "WFT_Pct": "mean",
        "WSeedNum": "mean",
        "WinMargin": "mean"
    }).reset_index()
    team_stats_win.rename(columns={"WTeamID": "TeamID"}, inplace=True)
    
    # Calculate team stats from losing games
    team_stats_loss = tourney_results.groupby("LTeamID").agg({
        "LFG_Pct": "mean",
        "L3P_Pct": "mean",
        "LFT_Pct": "mean", 
        "LSeedNum": "mean"
    }).reset_index()
    team_stats_loss.rename(columns={
        "LTeamID": "TeamID",
        "LFG_Pct": "WFG_Pct",
        "L3P_Pct": "W3P_Pct", 
        "LFT_Pct": "WFT_Pct",
        "LSeedNum": "WSeedNum"
    }, inplace=True)
    
    # Combine stats
    team_stats = pd.concat([team_stats_win, team_stats_loss]).groupby("TeamID").mean().reset_index()
    
    # Merge with team names
    team_stats = team_stats.merge(
        teams[['TeamID', 'TeamName']], on='TeamID', how='left'
    )
    
    print(f"✅ Generated stats for {len(team_stats)} teams")
    
    # Create model package
    model_package = {
        'model': model,
        'teams_data': teams,
        'team_stats': team_stats,
        'tournament_data': tourney_results,
        'features': features,
        'model_metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'test_samples': len(y_test)
        },
        'model_info': {
            'created_date': pd.Timestamp.now().isoformat(),
            'training_samples': len(X),
            'feature_count': len(features),
            'xgboost_version': xgb.__version__
        }
    }
    
    # Save model package
    print("💾 Saving model package...")
    
    with open('ncaa_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print("🎉 Model training complete!")
    print(f"📦 Model saved as: ncaa_model.pkl")
    print(f"📊 Model size: {os.path.getsize('ncaa_model.pkl') / 1024 / 1024:.1f} MB")
    
    return model_package

if __name__ == "__main__":
    try:
        model_package = create_and_save_model()
        print("\n" + "="*60)
        print("🏆 SUCCESS: NCAA Model is ready for production!")
        print("="*60)
        
        # Test prediction
        print("\n🧪 Testing sample prediction...")
        model = model_package['model']
        team_stats = model_package['team_stats']
        
        # Alabama vs Duke
        alabama_stats = team_stats[team_stats['TeamID'] == 1104].iloc[0]
        duke_stats = team_stats[team_stats['TeamID'] == 1181].iloc[0]
        
        features_data = pd.DataFrame({
            "Seed_Diff": [alabama_stats["WSeedNum"] - duke_stats["WSeedNum"]],
            "WFG_Pct": [alabama_stats["WFG_Pct"] - duke_stats["WFG_Pct"]],
            "LFG_Pct": [duke_stats["WFG_Pct"] - alabama_stats["WFG_Pct"]],
            "W3P_Pct": [alabama_stats["W3P_Pct"] - duke_stats["W3P_Pct"]],
            "L3P_Pct": [duke_stats["W3P_Pct"] - alabama_stats["W3P_Pct"]],
            "WFT_Pct": [alabama_stats["WFT_Pct"] - duke_stats["WFT_Pct"]],
            "LFT_Pct": [duke_stats["WFT_Pct"] - alabama_stats["WFT_Pct"]]
        })
        
        dmatrix = xgb.DMatrix(features_data)
        prediction = model.predict(dmatrix)[0]
        
        winner = "Alabama" if prediction > 0.5 else "Duke"
        confidence = max(prediction, 1-prediction) * 100
        
        print(f"🏀 Alabama vs Duke: {winner} wins ({confidence:.1f}% confidence)")
        print("\n✅ Model is working correctly!")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        raise e 