import os
import glob
import re

os.chdir(r'.\codes')

for file in glob.glob('*.py'):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all raw windows paths with standard os.path.join from a dynamic BASE_DIR
    
    # 1. Define dynamic base_dir
    dynamic_base = """import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))"""

    # If the file already has hardcoded base_dir, replace it
    content = re.sub(r'BASE_DIR\s*=\s*r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc"', 'BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))', content)
    content = re.sub(r'base_dir\s*=\s*r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc"', 'base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))', content)
    
    # Replace any leftover hardcoded paths
    content = re.sub(r'r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc\\datasets\\raw', r'os.path.join(BASE_DIR, "datasets", "raw"', content)
    content = re.sub(r'r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc\\datasets\\processed', r'os.path.join(BASE_DIR, "datasets", "processed"', content)
    content = re.sub(r'r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc\\plots/agent', r'os.path.join(BASE_DIR, "plots", "agent"', content)
    content = re.sub(r'r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc\\plots/xgboost', r'os.path.join(BASE_DIR, "plots", "xgboost"', content)
    content = re.sub(r'r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc\\plots', r'os.path.join(BASE_DIR, "plots"', content)
    content = re.sub(r'r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc\\results', r'os.path.join(BASE_DIR, "results"', content)
    content = re.sub(r'r"c:\\Users\\vashi\\OneDrive\\Desktop\\opensoc', r'BASE_DIR', content)
    
    # In some places, we used fr"c:\..." which is tricky. Let's fix those manually for safety
    content = content.replace(r'fr"plots/xgboost/', r'os.path.join(BASE_DIR, "plots", "xgboost", f"')
    content = content.replace(r'fr"plots/agent/', r'os.path.join(BASE_DIR, "plots", "agent", f"')
    content = content.replace(r'fr"plots/', r'os.path.join(BASE_DIR, "plots", f"')
    
    # Fix the missing parenthesis from replacing `r"..."` with `os.path.join(BASE_DIR, "...", "..."`
    # E.g. os.path.join(BASE_DIR, "datasets", "processed"\urbanev_ml_ready.csv" -> needs `)` and correct commas
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
