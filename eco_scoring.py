import pandas as pd
import json
import os
import numpy as np

def calculate_eco_scores():
    data_path = r"d:\hackthon\02_競賽數據_HINO\HINO_data.csv"
    output_path = r"d:\hackthon\02_競賽數據_HINO\eco_scores.js"

    print("Loading data...")
    # Load dataset
    df = pd.read_csv(data_path, on_bad_lines='skip')

    print("Processing features...")
    # Fill NA values to avoid errors
    df['gps.speed'] = df['gps.speed'].fillna(0)
    df['can.engine.rpm'] = df['can.engine.rpm'].fillna(0)
    df['can.engine.engineLoad'] = df['can.engine.engineLoad'].fillna(0)

    # Calculate boolean conditions for bad driving behaviors
    # Idling: Speed is 0 but RPM is > 0
    df['is_idling'] = (df['gps.speed'] == 0) & (df['can.engine.rpm'] > 0)
    # High RPM: Speed > 0 and RPM > 1800 (FMS Commercial Standard)
    df['is_high_rpm'] = (df['gps.speed'] > 0) & (df['can.engine.rpm'] > 1800)
    # Heavy Load: Engine Load > 80%
    df['is_heavy_load'] = df['can.engine.engineLoad'] > 80

    # Predictive Maintenance Anomalies
    # Cooling anomaly: high speed (>60), low load (<70), but high temp (>90)
    df['is_cooling_anomaly'] = (df['gps.speed'] > 60) & (df['can.engine.engineLoad'] < 70) & (df['can.engine.engineCoolantTemp'] > 90)
    # Transmission anomaly: high load (>90), high rpm (>2000), but low speed (<30)
    df['is_transmission_anomaly'] = (df['can.engine.engineLoad'] > 90) & (df['can.engine.rpm'] > 2000) & (df['gps.speed'] < 30)

    print("Aggregating by vehicle (enabledCode)...")
    # Group by vehicle ID
    grouped = df.groupby('enabledCode').agg(
        total_records=('time', 'count'),
        idling_count=('is_idling', 'sum'),
        high_rpm_count=('is_high_rpm', 'sum'),
        heavy_load_count=('is_heavy_load', 'sum'),
        cooling_anomaly_count=('is_cooling_anomaly', 'sum'),
        transmission_anomaly_count=('is_transmission_anomaly', 'sum'),
        avg_rpm=('can.engine.rpm', 'mean'),
        avg_speed=('gps.speed', 'mean')
    ).reset_index()

    # Calculate ratios
    grouped['idling_ratio'] = grouped['idling_count'] / grouped['total_records']
    grouped['high_rpm_ratio'] = grouped['high_rpm_count'] / grouped['total_records']
    grouped['heavy_load_ratio'] = grouped['heavy_load_count'] / grouped['total_records']

    print("Scoring (FMS Standard)...")
    # Base score is 100
    # Allowed Idling Baseline: 15% (0.15). Only penalize excess idling.
    grouped['excess_idling'] = np.maximum(0, grouped['idling_ratio'] - 0.15)
    
    # Peer Benchmarking Weights:
    # - Excess Idling Penalty: * 120
    # - Over-Revving (RPM > 1800) Penalty: * 150
    # - High Load (> 80%) Penalty: * 20
    grouped['score'] = 100.0 - (grouped['excess_idling'] * 120) - (grouped['high_rpm_ratio'] * 150) - (grouped['heavy_load_ratio'] * 20)
    
    # Cap score at 100 and floor at 0
    grouped['score'] = grouped['score'].clip(lower=0, upper=100).round(1)

    # Determine Grade
    def get_grade(score):
        if score >= 90: return 'A (優良)'
        elif score >= 80: return 'B (良好)'
        elif score >= 70: return 'C (普通)'
        elif score >= 60: return 'D (需改善)'
        else: return 'E (極度耗損)'
    
    grouped['grade'] = grouped['score'].apply(get_grade)

    # Generate personalized recommendations
    def generate_recommendation(row):
        recs = []
        if row['idling_ratio'] > 0.20:
            recs.append("怠速比例遠超同業基準 (15%)，建議減少停車未熄火時間")
        if row['high_rpm_ratio'] > 0.03:
            recs.append("轉速經常超過 1800 RPM，建議及早換檔維持綠色經濟轉速")
        if row['heavy_load_ratio'] > 0.2:
            recs.append("引擎經常處於高負載，請注意載重與爬坡檔位使用")
        
        if not recs:
            return "駕駛習慣良好，請繼續保持"
        return "；".join(recs)

    grouped['recommendation'] = grouped.apply(generate_recommendation, axis=1)

    # Convert to JSON serializable list of dicts
    results = []
    for _, row in grouped.iterrows():
        results.append({
            "vehicle_id": str(row['enabledCode']),
            "total_records": int(row['total_records']),
            "idling_ratio": round(float(row['idling_ratio']), 3),
            "high_rpm_ratio": round(float(row['high_rpm_ratio']), 3),
            "heavy_load_ratio": round(float(row['heavy_load_ratio']), 3),
            "score": float(row['score']),
            "grade": str(row['grade']),
            "recommendation": str(row['recommendation'])
        })
    
    # Sort by score descending
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Prepare maintenance alerts data
    maintenance_results = []
    for _, row in grouped.iterrows():
        cool = int(row['cooling_anomaly_count'])
        trans = int(row['transmission_anomaly_count'])
        if cool > 0 or trans > 0:
            maintenance_results.append({
                "vehicle_id": str(row['enabledCode']),
                "cooling_anomalies": cool,
                "transmission_anomalies": trans
            })
    # sort by total anomalies
    maintenance_results = sorted(maintenance_results, key=lambda x: x['cooling_anomalies'] + x['transmission_anomalies'], reverse=True)

    print(f"Exporting {len(results)} records to JS...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("const ecoScoresData = ")
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write(";\n\n")
        f.write("const maintenanceData = ")
        json.dump(maintenance_results, f, ensure_ascii=False, indent=2)
        f.write(";\n")
    
    print("Done! Data exported to", output_path)

if __name__ == "__main__":
    calculate_eco_scores()
