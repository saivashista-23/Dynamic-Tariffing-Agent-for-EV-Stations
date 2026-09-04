import subprocess
import sys

scripts = [
    "codes/1.acn_hourly_aggregation.py",
    "codes/2.process_urbanev.py",
    "codes/3.eda_plots.py",
    "codes/4.time_series_eda.py",
    "codes/5.engineer_ml_features.py",
    "codes/6.train_lightgbm.py",
    "codes/6.train_xgboost.py",
    "codes/7.pricing_monitoring_and_learning_agent.py"
]

for script in scripts:
    print(f"\\n{'='*50}\\nRunning {script}...\\n{'='*50}")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\\n[ERROR] IN {script}\\n")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"[SUCCESS] {script} completed successfully.")
        
print("\\n[ALL DONE] ALL SCRIPTS EXECUTED SUCCESSFULLY WITHOUT ERRORS!")
