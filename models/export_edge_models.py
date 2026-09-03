import numpy as np, pandas as pd, os, json, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import xgboost as xgb
np.random.seed(42); tf.random.set_seed(42)

m=pd.read_excel('data/cacak-monthly.xlsx').sort_values(['YEAR','MO']).reset_index(drop=True)
m['sin_m']=np.sin(2*np.pi*m.MO/12); m['cos_m']=np.cos(2*np.pi*m.MO/12)
m['t_idx']=np.arange(len(m)); m['ts']=m.YEAR*12+m.MO
FEAT=['ALLSKY_KT','ALLSKY_SRF_ALB','sin_m','cos_m','t_idx','ts']
TARG=['ALLSKY_SFC_SW_DWN','CLRSKY_SFC_SW_DWN']
X=m[FEAT].values.astype('float32'); Y=m[TARG].values.astype('float32')
sx,sy=MinMaxScaler(),MinMaxScaler(); Xs=sx.fit_transform(X); Ys=sy.fit_transform(Y)
sp=int(len(Xs)*0.8)

# ---------------- XGBoost ----------------
boosters=[]
for t in range(2):
    b=xgb.XGBRegressor(max_depth=3,n_estimators=40,learning_rate=0.25,
                       reg_lambda=1.0,base_score=0.5,random_state=42)
    b.fit(Xs[:sp],Ys[:sp,t]); boosters.append(b)
Pt=np.column_stack([b.predict(Xs[sp:]) for b in boosters])
r2_xgb=r2_score(sy.inverse_transform(Ys[sp:]).ravel(), sy.inverse_transform(Pt).ravel())

def flatten(b):
    df=b.get_booster().trees_to_dataframe(); flat=[]; offs=[]
    for _,g in df.groupby('Tree'):
        nodes={}
        for _,r in g.iterrows():
            nid=int(r.ID.split('-')[1])
            if r.Feature=='Leaf':
                nodes[nid]=(-1,float(r.Gain),0,0)
            else:
                nodes[nid]=(int(str(r.Feature)[1:]),float(r.Split),
                            int(r.Yes.split('-')[1]),int(r.No.split('-')[1]))
        offs.append(len(flat)); arr=[(0,0.0,0,0)]*(max(nodes)+1)
        for k,v in nodes.items(): arr[k]=v
        flat+=arr
    return flat,offs

def sim_xgb(flat,offs,x):
    s=0.5
    for off in offs:
        n=off
        while flat[n][0]>=0:
            n=off+(flat[n][2] if x[flat[n][0]]<flat[n][1] else flat[n][3])
        s+=flat[n][1]
    return s

FL=[flatten(b) for b in boosters]
probe=Xs[250]
for i,(fl,of) in enumerate(FL):
    a=sim_xgb(fl,of,probe); c=float(boosters[i].predict(probe.reshape(1,-1))[0])
    assert abs(a-c)<1e-5, f'export mismatch target {i}: {a} vs {c}'
print(f'XGBoost edge (depth 3, 40 trees x2): R2 = {r2_xgb:.4f}   [export verifikovan]')

# ---------------- ANN INT8 ----------------
W=12
Xw=np.array([Xs[i:i+W] for i in range(len(Xs)-W)]).reshape(len(Xs)-W,-1)
Yw=np.array([Ys[i+W] for i in range(len(Ys)-W)])
s2=int(len(Xw)*0.8)
ann=keras.Sequential([keras.layers.Input(shape=(Xw.shape[1],)),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(2,activation='linear')])
ann.compile(optimizer=keras.optimizers.Adam(0.01),loss='mse')
ann.fit(Xw[:s2],Yw[:s2],epochs=300,batch_size=16,verbose=0)
r2_ann=r2_score(sy.inverse_transform(Yw[s2:]).ravel(),
                sy.inverse_transform(ann.predict(Xw[s2:],verbose=0)).ravel())

L=[]
for lay in ann.layers:
    Wt,bs=lay.get_weights(); s=float(np.max(np.abs(Wt))/127.0)
    L.append(dict(w=np.clip(np.round(Wt/s),-127,127).astype(np.int8),s=s,
                  b=bs.astype(np.float32),shape=Wt.shape,
                  relu=lay.activation.__name__=='relu'))
def sim_ann(x):
    a=np.asarray(x,dtype=np.float64)
    for l in L:
        a=a@(l['w'].astype(np.float64)*l['s'])+l['b']
        if l['relu']: a=np.maximum(a,0)
    return a
r2_q=r2_score(sy.inverse_transform(Yw[s2:]).ravel(),
              sy.inverse_transform(np.array([sim_ann(r) for r in Xw[s2:]])).ravel())
print(f'ANN edge INT8 ({Xw.shape[1]}-32-16-2): R2 = {r2_q:.4f} (FP32 {r2_ann:.4f}), params={ann.count_params()}')

# ---------------- write header ----------------
def fl2c(v):
    t=f'{v:.9g}'
    if ('.' not in t) and ('e' not in t) and ('n' not in t): t+='.0'
    return t+'f'

def carr(t,name,vals,fmt):
    return f'static const {t} {name}[{len(vals)}] = {{'+','.join(fmt(v) for v in vals)+'};'
h=['// Auto-generated edge models for the ESP32 benchmark. Do not edit by hand.',
   '#include <stdint.h>','',f'#define NFEAT {len(FEAT)}','']
for i,(fl,of) in enumerate(FL):
    p=f'xgb_t{i}'
    h+=[carr('int16_t',p+'_feat',[a[0] for a in fl],lambda v:str(v)),
        carr('float',  p+'_thr', [a[1] for a in fl],fl2c),
        carr('uint16_t',p+'_yes',[a[2] for a in fl],lambda v:str(v)),
        carr('uint16_t',p+'_no', [a[3] for a in fl],lambda v:str(v)),
        carr('uint16_t',p+'_off',of,lambda v:str(v)),
        f'#define XGB_T{i}_NTREES {len(of)}','']
h+=['static const float xgb_base = 0.5f;','',
    f'#define ANN_IN {Xw.shape[1]}',f'#define ANN_NL {len(L)}','']
for i,l in enumerate(L):
    h+=[carr('int8_t',f'ann_w{i}',l['w'].T.ravel().tolist(),lambda v:str(int(v))),
        carr('float', f'ann_b{i}',l['b'].tolist(),fl2c),
        f'static const float ann_s{i} = {l["s"]:.9e}f;',
        f'#define ANN_L{i}_IN {l["shape"][0]}',
        f'#define ANN_L{i}_OUT {l["shape"][1]}','']
open('edge/esp32_bench/edge_models.h','w').write('\n'.join(h)+'\n')

np.savetxt('edge/esp32_bench/ref_in.txt',
           np.concatenate([probe,Xw[250-W]]).reshape(1,-1),fmt='%.9f')
ref=dict(xgb=[float(boosters[i].predict(probe.reshape(1,-1))[0]) for i in range(2)],
         ann=[float(v) for v in sim_ann(Xw[250-W])],
         r2_xgb=r2_xgb,r2_ann_fp32=r2_ann,r2_ann_int8=r2_q,
         ann_params=int(ann.count_params()),xgb_nodes=sum(len(f) for f,_ in FL))
json.dump(ref,open('edge/esp32_bench/edge_ref.json','w'),indent=1)
print('\nreferentni izlazi:',{k:[round(v,6) for v in ref[k]] for k in ['xgb','ann']})
print('header zapisan:',os.path.getsize('edge/esp32_bench/edge_models.h'),'bajtova')
