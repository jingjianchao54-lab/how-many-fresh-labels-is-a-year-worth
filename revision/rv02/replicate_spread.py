"""Describe five existing training draws. Read predictions only; no bootstrap or refit."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def score(y, pred):
    cm = np.bincount(y * 10 + pred.astype(int), minlength=100).reshape(10,10)
    return (2 * cm.diagonal() / (cm.sum(0) + cm.sum(1))).mean()

def crossing(sizes, fresh, stale):
    if fresh[0] >= stale:
        return 0.0
    for i in range(1,len(sizes)):
        if fresh[i] >= stale:
            fraction=(stale-fresh[i-1])/(fresh[i]-fresh[i-1])
            return float(np.exp(np.log(sizes[i-1])+fraction*np.log(sizes[i]/sizes[i-1])))
    return float("inf")

def run():
    results={}
    for ds in ("reddit","stackexchange"):
        frozen=json.loads((HERE/"scored"/f"results_{ds}_RV02.json").read_text())
        meta=frozen["meta"]
        years,sizes=meta["years"],meta["sizes"]
        N=sizes[-1]
        classes=json.loads((ROOT/"results/gate.json").read_text())[ds]["decision"]["classes"]
        with gzip.open(ROOT/"results"/f"sample_index_{ds}.csv.gz","rt") as stream:
            reader=csv.reader(stream)
            next(reader)
            y=np.array([classes.index(row[0]) for row in reader if int(row[-2])==years[-1] and int(row[-1])==2])
        assert len(y)==10*meta["eval_items_per_cell"]
        for model in ("M6G","M6L"):
            folder=HERE/"work/runs"/ds/"test"/model
            rs,curves,old=[],[],[]
            for r in range(1,6):
                def pred(exp,yr,n,idx):
                    with np.load(folder/f"{exp}_{r}_{yr}_{n}_None.npz") as z:
                        return score(y,z["pred"][idx])
                stale=pred("e1",years[0],N,-1)
                fresh=[pred("e2",years[-1],n,0) for n in sizes[:-1]]+[pred("e1",years[-1],N,-1)]
                rs.append(crossing(sizes,fresh,stale)/N)
                curves.append(fresh);old.append(stale)
            pooled=crossing(sizes,np.mean(curves,axis=0),np.mean(old))/N
            assert abs(pooled-frozen["models"][model]["exchange"]["6"]["rho"])<0.00006
            results[f"{ds}_{model}"]={"rho_6_per_draw":rs,"sample_sd":float(np.std(rs,ddof=1)),"pooled_check":pooled}
    return results

if __name__=="__main__":
    print(json.dumps(run(),indent=2))

