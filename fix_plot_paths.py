import os

os.chdir(r'c:\Users\vashi\OneDrive\Desktop\opensoc\codes')

# 3.eda_plots.py -> plots/eda_demand_trends/
with open('3.eda_plots.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(r'plots/', r'plots/eda_demand_trends/')
content = content.replace("os.makedirs('plots', exist_ok=True)", "os.makedirs('plots/eda_demand_trends', exist_ok=True)")
content = content.replace("os.makedirs('plots/eda_demand_trends/agent'", "os.makedirs('plots/agent'") # just in case

with open('3.eda_plots.py', 'w', encoding='utf-8') as f:
    f.write(content)

# 4.time_series_eda.py -> plots/edaforpredictionmodel/
with open('4.time_series_eda.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(r'plots/', r'plots/edaforpredictionmodel/')
content = content.replace("os.makedirs('plots', exist_ok=True)", "os.makedirs('plots/edaforpredictionmodel', exist_ok=True)")

with open('4.time_series_eda.py', 'w', encoding='utf-8') as f:
    f.write(content)

# 6.train_lightgbm.py -> plots/lightgbm/
with open('6.train_lightgbm.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(r'plots/{target}_importance.png', r'plots/lightgbm/{target}_importance.png')
content = content.replace("os.makedirs('plots/models', exist_ok=True)", "os.makedirs('plots/lightgbm', exist_ok=True)")

with open('6.train_lightgbm.py', 'w', encoding='utf-8') as f:
    f.write(content)
