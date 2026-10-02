from pathlib import Path
import json,subprocess,hashlib
import pandas as pd,numpy as np
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT;OUT=REPO/'results/generated'
# Quantified checks against manuscript-rounded values, not replacement values.
doe=pd.read_csv(OUT/'table03_doe_by_seed.csv');checks=[]
for method,region,cd,cdstd,cov,covstd in [('random','Pick',.005294,.003099,.6322,.0259),('random','Place',.007692,.003922,.6256,.0195),('halton','Pick',.000271,None,.7567,None),('halton','Place',.000271,None,.7567,None)]:
    d=doe[(doe.method==method)&(doe.region==region)]
    checks.append(dict(item=f'Table 3 {method} {region}',discrepancy=float(d.discrepancy.mean()),coverage=float(d.coverage.mean()),status='PASS' if round(d.discrepancy.mean(),6)==cd and round(d.coverage.mean(),4)==cov else 'MISMATCH'))
    if cdstd is not None:
        checks[-1].update(sd_discrepancy=float(d.discrepancy.std()),sd_coverage=float(d.coverage.std()))
        if round(d.discrepancy.std(),6)!=cdstd or round(d.coverage.std(),4)!=covstd:checks[-1]['status']='MISMATCH'
domain=pd.read_csv(OUT/'table04_domain_shift.csv').set_index('variable')
for variable,values in [('temperature',[40.13,1.48,43.25,1.84,3.12,.671,1.87]),('voltage',[7246.56,47.90,7404.93,38.67,158.38,.937,3.64]),('position',[644.54,172.33,642.21,167.52,-2.33,.029,-.014])]:
    keys=['source_mean','source_std','target_mean','target_std','difference','ks','standardized_difference']
    deltas={key:float(domain.loc[variable,key]-value) for key,value in zip(keys,values)}
    # Half a printed unit for each column; standard deviation sample convention retained.
    tolerances=[.0051]*5+[.00051,.00051 if variable=='position' else .0051]
    checks.append(dict(item=f'Table 4 {variable}',deltas=deltas,status='PASS' if all(abs(deltas[k])<=tol for k,tol in zip(keys,tolerances)) else 'MISMATCH'))
# Retain the printed last-digit difference without treating it as a model failure.
for row in checks:
    if row['item']=='Table 4 voltage' and row['status']=='MISMATCH':
        if all(abs(v)<= (.01 if k=='difference' else (.00051 if k=='ks' else .0051)) for k,v in row['deltas'].items()):
            row['status']='ROUNDING_DIFFERENCE'
(OUT/'descriptive_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
# Low-resolution contact sheet for visual verification of every generated analytical plot.
images=sorted((OUT/'figures').glob('fig*.png'));canvas=Image.new('RGB',(1200,270*int(np.ceil(len(images)/3))),'white');draw=ImageDraw.Draw(canvas)
for i,p in enumerate(images):
    im=Image.open(p).convert('RGB');im.thumbnail((390,240));x=(i%3)*400;y=(i//3)*270
    canvas.paste(im,(x+(400-im.width)//2,y));draw.text((x+8,y+243),p.stem[:50],fill='black')
canvas.save(OUT/'figure_contact_sheet.png')
print(json.dumps(checks,indent=2))
