import os, numpy as np, pandas as pd, random
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED=42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

m = pd.read_excel('data/cacak-monthly.xlsx')
m = m.sort_values(['YEAR','MO']).reset_index(drop=True)

# Feature set from the study: ASICI + ASSA as predictors, plus calendar-based transformations.m['sin_m'] = np.sin(2*np.pi*m.MO/12)
m['cos_m'] = np.cos(2*np.pi*m.MO/12)
m['t_idx'] = np.arange(len(m))
m['ts']    = m.YEAR*12 + m.MO

FEAT = ['ALLSKY_KT','ALLSKY_SRF_ALB','sin_m','cos_m','t_idx','ts']
TARG = ['ALLSKY_SFC_SW_DWN','CLRSKY_SFC_SW_DWN']

X = m[FEAT].values.astype('float32')
Y = m[TARG].values.astype('float32')

sx, sy = MinMaxScaler(), MinMaxScaler()
Xs, Ys = sx.fit_transform(X), sy.fit_transform(Y)

# --- 12-month sliding windows, as used for sequential architectures. ---
W = 12
Xw = np.array([Xs[i:i+W] for i in range(len(Xs)-W)])
Yw = np.array([Ys[i+W]   for i in range(len(Ys)-W)])

split = int(len(Xw)*0.8)           
Xtr, Xte = Xw[:split], Xw[split:]
Ytr, Yte = Yw[:split], Yw[split:]
print(f'train {len(Xtr)}  test {len(Xte)} (test = poslednjih {len(Xte)} meseci)')

model = keras.Sequential([
    keras.layers.Input(shape=(W, len(FEAT))),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'), keras.layers.Dropout(0.2),
    keras.layers.Dense(64,  activation='relu'), keras.layers.Dropout(0.2),
    keras.layers.Dense(32,  activation='relu'), keras.layers.Dropout(0.2),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16), keras.layers.LeakyReLU(),
    keras.layers.Dense(2, activation='linear'),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
model.fit(Xtr, Ytr, epochs=100, batch_size=16, verbose=0)
print('parametara:', model.count_params())

P = model.predict(Xte, verbose=0)
a = sy.inverse_transform(Yte); b = sy.inverse_transform(P)

af, bf = a.ravel(), b.ravel()
mae  = mean_absolute_error(af, bf)
mape = np.mean(np.abs((af-bf)/af))*100
mse  = mean_squared_error(af, bf)
rmse = np.sqrt(mse)
msle = np.mean((np.log1p(np.clip(af,0,None))-np.log1p(np.clip(bf,0,None)))**2)
r2   = r2_score(af, bf)
n, j = len(af), len(FEAT)
adj  = 1-(1-r2)*(n-1)/(n-j-1)

print()
print(f'{"MAE":>8}{"MAPE%":>9}{"MSE":>9}{"RMSE":>9}{"MSLE":>9}{"RMSLE":>9}{"R2":>9}{"AdjR2":>9}')
print(f'{mae:>8.4f}{mape:>9.2f}{mse:>9.4f}{rmse:>9.4f}{msle:>9.4f}{np.sqrt(msle):>9.4f}{r2:>9.4f}{adj:>9.4f}')
print()
for i,t in enumerate(TARG):
    print(f'  {t}: R2 = {r2_score(a[:,i], b[:,i]):.4f}')
