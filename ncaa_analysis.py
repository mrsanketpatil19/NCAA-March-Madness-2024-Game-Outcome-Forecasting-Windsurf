#!/usr/bin/env python3
"""
NCAA Tournament Data Analysis & Insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data():
    """Load NCAA tournament data"""
    base_path = "./march-machine-learning-mania-2025/"
    
    # Load key datasets
    results = pd.read_csv(os.path.join(base_path, "MNCAATourneyDetailedResults.csv"))
    seeds = pd.read_csv(os.path.join(base_path, "MNCAATourneySeeds.csv"))
    teams = pd.read_csv(os.path.join(base_path, "MTeams.csv"))
    
    # Merge data
    results = results.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how="left")
    results = results.rename(columns={'Seed': 'WSeed'}).drop(columns=['TeamID'])
    
    results = results.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how="left")
    results = results.rename(columns={'Seed': 'LSeed'}).drop(columns=['TeamID'])
    
    results = results.merge(teams[['TeamID', 'TeamName']], left_on="WTeamID", right_on="TeamID", how="left")
    results = results.rename(columns={'TeamName': 'WTeamName'}).drop(columns=['TeamID'])
    
    results = results.merge(teams[['TeamID', 'TeamName']], left_on="LTeamID", right_on="TeamID", how="left")
    results = results.rename(columns={'TeamName': 'LTeamName'}).drop(columns=['TeamID'])
    
    # Extract seed numbers
    results["WSeedNum"] = results["WSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
    results["LSeedNum"] = results["LSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
    results["WinMargin"] = results["WScore"] - results["LScore"]
    
    return results

def analyze_seed_performance(data):
    """Analyze how seeds perform historically"""
    print("📊 SEED PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Calculate win percentage by seed
    seed_wins = data.groupby('WSeedNum').size()
    seed_total = data.groupby('WSeedNum').size() + data.groupby('LSeedNum').size()
    
    win_pct = (seed_wins / seed_total * 100).fillna(0)
    
    print("🏆 Win Percentage by Seed:")
    for seed in sorted(win_pct.index):
        if not pd.isna(seed):
            print(f"   Seed {int(seed):2d}: {win_pct[seed]:.1f}%")
    
    # Plot seed performance
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(win_pct.index, win_pct.values, color='lightblue', edgecolor='navy')
    plt.xlabel('Seed Number')
    plt.ylabel('Win Percentage (%)')
    plt.title('Historical Win Percentage by Seed')
    plt.grid(axis='y', alpha=0.3)
    
    return win_pct

def analyze_upsets(data):
    """Analyze tournament upsets"""
    print("\n🔥 UPSET ANALYSIS")
    print("=" * 50)
    
    # Calculate seed differences (positive = upset)
    data['SeedDiff'] = data['LSeedNum'] - data['WSeedNum']
    upsets = data[data['SeedDiff'] > 0]
    
    print(f"📈 Total Upsets: {len(upsets)} out of {len(data)} games ({len(upsets)/len(data)*100:.1f}%)")
    
    # Biggest upsets
    biggest_upsets = upsets.nlargest(10, 'SeedDiff')[['Season', 'WTeamName', 'WSeedNum', 'LTeamName', 'LSeedNum', 'WScore', 'LScore']]
    print("\n🚨 Top 10 Biggest Upsets:")
    for _, upset in biggest_upsets.iterrows():
        print(f"   {int(upset['Season'])}: #{int(upset['WSeedNum'])} {upset['WTeamName']} beat #{int(upset['LSeedNum'])} {upset['LTeamName']} ({int(upset['WScore'])}-{int(upset['LScore'])})")
    
    # Plot upset distribution
    plt.subplot(1, 2, 2)
    upset_counts = upsets['SeedDiff'].value_counts().sort_index()
    plt.bar(upset_counts.index, upset_counts.values, color='salmon', edgecolor='darkred')
    plt.xlabel('Seed Difference')
    plt.ylabel('Number of Upsets')
    plt.title('Distribution of Upsets by Seed Difference')
    plt.grid(axis='y', alpha=0.3)
    
    return upsets

def analyze_champions(data):
    """Analyze championship patterns"""
    print("\n🏆 CHAMPIONSHIP ANALYSIS")
    print("=" * 50)
    
    # Find champions (assuming they're the teams that win the most games each season)
    champions = []
    for season in data['Season'].unique():
        season_data = data[data['Season'] == season]
        wins = season_data['WTeamName'].value_counts()
        if len(wins) > 0:
            champion = wins.index[0]
            # Find champion's seed
            champ_seed = season_data[season_data['WTeamName'] == champion]['WSeedNum'].iloc[0] if len(season_data[season_data['WTeamName'] == champion]) > 0 else None
            champions.append({'Season': season, 'Champion': champion, 'Seed': champ_seed})
    
    champions_df = pd.DataFrame(champions)
    champions_df = champions_df.dropna()
    
    print(f"📅 Analyzed {len(champions_df)} championship seasons")
    
    # Seed distribution of champions
    seed_counts = champions_df['Seed'].value_counts().sort_index()
    print("\n🥇 Championships by Seed:")
    for seed, count in seed_counts.items():
        print(f"   Seed {int(seed):2d}: {count} championships")
    
    # Most successful programs
    program_counts = champions_df['Champion'].value_counts().head(10)
    print(f"\n🏛️ Most Successful Programs:")
    for team, count in program_counts.items():
        print(f"   {team}: {count} championships")
    
    return champions_df

def analyze_scoring_trends(data):
    """Analyze scoring trends over time"""
    print("\n📈 SCORING TRENDS")
    print("=" * 50)
    
    # Calculate average scores by season
    yearly_scores = data.groupby('Season').agg({
        'WScore': 'mean',
        'LScore': 'mean',
        'WinMargin': 'mean'
    }).reset_index()
    
    yearly_scores['TotalScore'] = yearly_scores['WScore'] + yearly_scores['LScore']
    
    recent_avg = yearly_scores[yearly_scores['Season'] >= 2020]['TotalScore'].mean()
    all_time_avg = yearly_scores['TotalScore'].mean()
    
    print(f"🎯 Average Total Score (All Time): {all_time_avg:.1f}")
    print(f"🎯 Average Total Score (2020+): {recent_avg:.1f}")
    print(f"📊 Average Win Margin: {yearly_scores['WinMargin'].mean():.1f} points")
    
    # Plot scoring trends
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(yearly_scores['Season'], yearly_scores['TotalScore'], marker='o', color='green')
    plt.xlabel('Season')
    plt.ylabel('Average Total Score')
    plt.title('Scoring Trends Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(yearly_scores['Season'], yearly_scores['WinMargin'], marker='s', color='orange')
    plt.xlabel('Season')
    plt.ylabel('Average Win Margin')
    plt.title('Win Margin Trends')
    plt.grid(True, alpha=0.3)
    
    # Calculate shooting efficiency
    data['WFG_Pct'] = data['WFGM'] / data['WFGA']
    data['W3P_Pct'] = data['WFGM3'] / data['WFGA3']
    
    efficiency_trends = data.groupby('Season').agg({
        'WFG_Pct': 'mean',
        'W3P_Pct': 'mean'
    }).reset_index()
    
    plt.subplot(2, 2, 3)
    plt.plot(efficiency_trends['Season'], efficiency_trends['WFG_Pct']*100, marker='o', label='FG%', color='blue')
    plt.plot(efficiency_trends['Season'], efficiency_trends['W3P_Pct']*100, marker='s', label='3P%', color='red')
    plt.xlabel('Season')
    plt.ylabel('Shooting Percentage')
    plt.title('Shooting Efficiency Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    data['TotalScore'] = data['WScore'] + data['LScore']
    plt.hist(data['TotalScore'], bins=30, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    plt.xlabel('Total Score (Both Teams)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Game Scores')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return yearly_scores

def main():
    """Main analysis function"""
    print("🏀 NCAA TOURNAMENT DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    data = load_data()
    print(f"📊 Loaded {len(data)} tournament games from {data['Season'].min()}-{data['Season'].max()}")
    
    # Run analyses
    seed_performance = analyze_seed_performance(data)
    upsets = analyze_upsets(data)
    champions = analyze_champions(data)
    scoring_trends = analyze_scoring_trends(data)
    
    # Save plots
    plt.savefig('ncaa_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Analysis plots saved as 'ncaa_analysis.png'")
    
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main() 