import os
import shutil
import glob
import json

base = r'c:\Users\vashi\OneDrive\Desktop\opensoc\plots'

# 1. Move plots/lightgbm to plots/models/lightgbm
try:
    os.makedirs(os.path.join(base, 'models', 'lightgbm'), exist_ok=True)
    for f in glob.glob(os.path.join(base, 'lightgbm', '*.*')):
        shutil.move(f, os.path.join(base, 'models', 'lightgbm'))
    os.rmdir(os.path.join(base, 'lightgbm'))
except Exception as e:
    pass # Ignore if already moved

# 2. Update main_colab.ipynb glob patterns
notebook_path = r'c:\Users\vashi\OneDrive\Desktop\opensoc\main_colab.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        
        # Update Step 3
        if 'codes/3.eda_plots.py' in source:
            source = source.replace("glob.glob('plots/*.png')", "glob.glob('plots/eda_demand_trends/*.png')")
            
        # Update Step 6
        if 'codes/6.train_xgboost.py' in source:
            source = source.replace("glob.glob('plots/xgboost/*.png')", "glob.glob('plots/models/*/*.png')")
            
        cell['source'] = [line + '\n' for line in source.strip().split('\n')]
            
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Updated main_colab.ipynb successfully.')
