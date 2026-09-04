import os
import re
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

md_file = r'C:\Users\vashi\.gemini\antigravity\brain\afc76427-bdf8-4c70-a05e-ef2bcb9da2c2\OpenSOC_Project_Summary.md'
out_file = r'c:\Users\vashi\OneDrive\Desktop\opensoc\OpenSOC_Project_Summary.docx'

doc = Document()

with open(md_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

def parse_inline(p, text):
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = p.add_run(part[2:-2])
            run.bold = True
        else:
            sub_parts = re.split(r'(`.*?`)', part)
            for sp in sub_parts:
                if sp.startswith('`') and sp.endswith('`'):
                    run = p.add_run(sp[1:-1])
                    run.font.name = 'Consolas'
                else:
                    p.add_run(sp)

for line in lines:
    line = line.strip()
    if not line or line == '---':
        continue
        
    if line.startswith('# '):
        p = doc.add_heading(level=1)
        parse_inline(p, line[2:])
    elif line.startswith('## '):
        p = doc.add_heading(level=2)
        parse_inline(p, line[3:])
    elif line.startswith('### '):
        p = doc.add_heading(level=3)
        parse_inline(p, line[4:])
    elif line.startswith('* '):
        p = doc.add_paragraph(style='List Bullet')
        parse_inline(p, line[2:])
    elif line.startswith('- '):
        p = doc.add_paragraph(style='List Bullet 2')
        parse_inline(p, line[2:])
    else:
        p = doc.add_paragraph()
        parse_inline(p, line)

doc.save(out_file)
print('Word document generated at', out_file)
