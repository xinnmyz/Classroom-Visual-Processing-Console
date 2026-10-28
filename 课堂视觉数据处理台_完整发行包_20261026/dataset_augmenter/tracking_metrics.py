from __future__ import annotations
"""Tracking metrics: IDF1, MOTA, simplified HOTA and occlusion recovery."""
import json, math
from pathlib import Path
from typing import Any, Sequence
import pandas as pd


def _col(df, *names):
    m={str(c).lower().strip():c for c in df.columns}
    for n in names:
        if n.lower() in m:return m[n.lower()]
    return None

def _num(x):
    try:
        v=float(x); return v if math.isfinite(v) else float("nan")
    except (TypeError,ValueError): return float("nan")

def _box(row,prefix=""):
    p=(prefix+"_") if prefix else ""
    fs=[p+"x1",p+"y1",p+"x2",p+"y2"]
    if not all(f in row.index for f in fs): return None
    v=tuple(_num(row[f]) for f in fs)
    return v if all(math.isfinite(x) for x in v) else None

def _iou(a,b):
    if a is None or b is None:return 0.0
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); u=aa+bb-inter
    return inter/u if u>0 else 0.0

def _prep(df, truth=False):
    if df is None:return pd.DataFrame()
    x=df.copy(); idc=_col(x,"gt_person_id","ground_truth_id","track_id","person_id") if truth else _col(x,"person_id","pred_person_id","track_id")
    x["_id"]=x[idc].astype(str) if idc else ""
    x["_asset"] = x.apply(
        lambda r: str(r.get("asset_id", r.get("video_id", r.get("file_name", "all")))) or "all",
        axis=1,
    )
    x["_frame"] = x.apply(
        lambda r: str(r.get("frame_index", r.get("frame", r.get("timestamp_seconds", "0")))),
        axis=1,
    )
    x["_box"]=x.apply(lambda r:_box(r,"gt") if truth else _box(r,""),axis=1)
    x["_occluded"]=x.apply(lambda r:str(r.get("occluded",r.get("is_occluded",""))).lower() in {"1","true","yes","y"},axis=1)
    return x

def _group_eval(pred,gt,threshold):
    tp=fp=fn=ids=0; prev={}; assoc=0
    frames=sorted(set(zip(pred._asset,pred._frame))|set(zip(gt._asset,gt._frame)))
    for asset,frame in frames:
        p=pred[(pred._asset==asset)&(pred._frame==frame)]; g=gt[(gt._asset==asset)&(gt._frame==frame)]
        pairs=[]
        for pi,pr in p.iterrows():
            for gi,gr in g.iterrows():
                s=_iou(pr._box,gr._box)
                if s>=threshold:pairs.append((s,pi,gi))
        usedp=set();usedg=set()
        for s,pi,gi in sorted(pairs,reverse=True):
            if pi in usedp or gi in usedg:continue
            usedp.add(pi);usedg.add(gi);tp+=1
            pid,gid=str(p.loc[pi,"_id"]),str(g.loc[gi,"_id"])
            if gid in prev and prev[gid]!=pid:ids+=1
            elif gid not in prev:assoc+=1
            prev[gid]=pid
        fp+=len(p)-len(usedp);fn+=len(g)-len(usedg)
    gt_total=tp+fn; idf1=2*tp/max(1,2*tp+fp+fn); mota=1-(fp+fn+ids)/max(1,gt_total)
    deta=tp/max(1e-9,tp+.5*(fp+fn)); assa=max(0,tp-ids)/max(1,tp); hota=math.sqrt(max(0,deta*assa))
    return {"gt_count":gt_total,"pred_count":tp+fp,"tp":tp,"fp":fp,"fn":fn,"id_switches":ids,"IDF1":idf1,"MOTA":mota,"DetA":deta,"AssA":assa,"HOTA_simplified":hota}

def _recovery(pred,gt,threshold):
    if not gt._occluded.any():return float("nan")
    opp=ok=0
    for (asset,gid),g in gt[gt._occluded].groupby(["_asset","_id"]):
        occ=set(g._frame.astype(str)); allg=gt[(gt._asset==asset)&(gt._id==gid)].sort_values("_frame")
        for _,r in allg.iterrows():
            if str(r._frame) not in occ and any(str(f)<str(r._frame) for f in occ):
                opp+=1; p=pred[(pred._asset==asset)&(pred._frame==r._frame)]
                ok+=any(_iou(r._box,pr._box)>=threshold for _,pr in p.iterrows()); break
    return ok/opp if opp else float("nan")

def evaluate_tracking(predictions:pd.DataFrame,ground_truth:pd.DataFrame,*,iou_threshold=.5,group_cols:Sequence[str]=("asset_id",)):
    p,g=_prep(predictions),_prep(ground_truth,True)
    if p.empty and g.empty:return pd.DataFrame()
    rows=[]; groups=[c for c in group_cols if c in predictions.columns and c in ground_truth.columns]
    keys=set(p._asset)|set(g._asset) if not groups else set(map(tuple,p[groups].drop_duplicates().itertuples(index=False,name=None)))|set(map(tuple,g[groups].drop_duplicates().itertuples(index=False,name=None)))
    for key in sorted(keys,key=str):
        kk=key if isinstance(key,tuple) else (key,); pp=p;gg=g
        for c,v in zip(groups or ["_asset"],kk): pp=pp[pp[c].astype(str)==str(v)];gg=gg[gg[c].astype(str)==str(v)]
        row=_group_eval(pp,gg,iou_threshold);row["occlusion_recovery_rate"]=_recovery(pp,gg,iou_threshold);row.update({c:v for c,v in zip(groups or ["asset_id"],kk)});rows.append(row)
    row=_group_eval(p,g,iou_threshold);row["occlusion_recovery_rate"]=_recovery(p,g,iou_threshold);row["group"]="__overall__";rows.append(row)
    return pd.DataFrame(rows)

def write_tracking_metrics_report(predictions,ground_truth,output_dir,*,iou_threshold=.5):
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);m=evaluate_tracking(predictions,ground_truth,iou_threshold=iou_threshold)
    csv=root/"tracking_metrics.csv";xlsx=root/"tracking_metrics.xlsx";js=root/"tracking_metrics.json";m.to_csv(csv,index=False,encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx,engine="openpyxl") as w:m.to_excel(w,sheet_name="tracking_metrics",index=False)
    js.write_text(json.dumps(m.to_dict("records"),ensure_ascii=False,indent=2,default=str),encoding="utf-8");return {"csv":csv,"xlsx":xlsx,"json":js}

if __name__=="__main__":
 p=pd.DataFrame([{"asset_id":"v","frame_index":0,"person_id":1,"x1":0,"y1":0,"x2":10,"y2":10}]);g=pd.DataFrame([{"asset_id":"v","frame_index":0,"track_id":7,"gt_x1":0,"gt_y1":0,"gt_x2":10,"gt_y2":10}]);print(evaluate_tracking(p,g))
