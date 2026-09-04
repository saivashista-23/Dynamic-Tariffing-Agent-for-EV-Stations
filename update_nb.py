import json

notebook_path = r'c:\Users\vashi\OneDrive\Desktop\opensoc\main_colab.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source_str = ''.join(cell.get('source', []))
        
        if 'codes/4.time_series_eda.py' in source_str and 'glob' not in source_str:
            new_source = """!python codes/4.time_series_eda.py

from IPython.display import Image, display
import glob

print("Time Series EDA Plots:")
for img in glob.glob('plots/edaforpredictionmodel/*.png'):
    print(img)
    display(Image(filename=img))
"""
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Updated main_colab.ipynb Step 4.')
