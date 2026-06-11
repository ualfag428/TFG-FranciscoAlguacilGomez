import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import random
import os

SEED = 50
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


tfd = tfp.distributions

# 1. CARGA DE DATOS

df = pd.read_csv(r'C:\Users\pacoa\Documents\Universidad-Grado en Matematicas\CUARTO CURSO\TFG\Python\datos_climaticos07.txt')

features = [
    'temperatura_c',        # 0
    'presion_hpa',          # 1
    'humedad_relativa',     # 2
    'velocidad_viento_10m', # 3
    'radiacion_ghi'         # 4
]

n_features = len(features)
data = df[features].values.astype(np.float32)


# 2. CREACION DE VENTANAS - TRAIN/TEST - ANTI LEAKAGE

window_size = 24
horizon = 1

total_windows = len(data) - window_size - horizon + 1
test_size  = int(total_windows * 0.2)
train_size = total_windows - test_size

train_cut_index = train_size + window_size - 1
scaler = MinMaxScaler()
scaler.fit(data[:train_cut_index])
scaled_data = scaler.transform(data)

def create_windows_multi(dataset, window_size=24, horizon=1, salto=1):
    X, y = [], []
    for i in range(0, len(dataset) - window_size - horizon + 1, salto):
        X.append(dataset[i:(i + window_size), :])
        target_idx = i + window_size + horizon - 1
        y.append(dataset[target_idx, 0:5])
    return np.array(X), np.array(y)

X, y = create_windows_multi(scaled_data, window_size, horizon, salto=1)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# 3. POSITIONAL ENCODING PARA TRANSFORMER

class PositionalEncoding(layers.Layer):
    def __init__(self, max_steps, max_dims, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        if max_dims % 2 == 1: max_dims += 1
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2]  = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_encoding = tf.constant(pos_emb, dtype=tf.float32)

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_encoding[:, :shape[1], :shape[2]]


# 4. MODELO (LSTM + TRANSFORMER + FLIPOUT)

input_layer = layers.Input(shape=(window_size, n_features))

lstm_branch = layers.LSTM(64, return_sequences=False, dropout=0.2)(input_layer)

trans_branch = layers.Dense(128)(input_layer)
trans_branch = PositionalEncoding(max_steps=window_size, max_dims=128)(trans_branch)
trans_branch = layers.LayerNormalization(epsilon=1e-6)(trans_branch)
attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(trans_branch, trans_branch)
trans_branch = layers.Add()([trans_branch, attention_out])
trans_branch = layers.GlobalMaxPooling1D()(trans_branch)
trans_branch = layers.Dense(128, activation="relu")(trans_branch)

combined = layers.Concatenate()([lstm_branch, trans_branch])


N_train = len(X_train)
kl_factor = 1.0 / N_train
kl_divergence_function = lambda q, p, _: kl_factor * tfd.kl_divergence(q, p)

combined = layers.Dense(128, activation='relu')(combined)
combined = layers.Dropout(0.25)(combined)  
combined = tfp.layers.DenseFlipout(64, activation='relu', kernel_divergence_fn=kl_divergence_function)(combined)
combined = tfp.layers.DenseFlipout(32, activation='relu', kernel_divergence_fn=kl_divergence_function)(combined)

params = layers.Dense(tfp.layers.IndependentNormal.params_size(5))(combined)


output_layer = tfp.layers.DistributionLambda(
    lambda t: tfd.Independent(
        tfd.Normal(
            loc=t[..., :5],
            scale=0.02 + tf.math.softplus(t[..., 5:] * 0.3)  
        ),
        reinterpreted_batch_ndims=1
    )
)(params)

model = models.Model(inputs=input_layer, outputs=output_layer)

# Funcion de perdida personalizada    

def crear_loss_bayesiana(indices_mae=[], indices_mse=[],
                         pesos_mae=None, pesos_mse=None,
                         peso_nll=0.05, dinamico=True,
                         factor_sensibilidad=5.0):

    def loss_personalizada(y_true, y_pred):
        mu = y_pred.mean()
        loss_total = 0.0

        if len(indices_mae) > 0:
            mae_vars = tf.abs(
                tf.gather(y_true, indices_mae, axis=1) -
                tf.gather(mu,     indices_mae, axis=1)
            )
            if pesos_mae is not None:
                w = tf.constant(pesos_mae, dtype=tf.float32)
                mae_vars = mae_vars * w
            loss_total += tf.reduce_mean(mae_vars)

        if len(indices_mse) > 0:
            mse_vars = tf.square(
                tf.gather(y_true, indices_mse, axis=1) -
                tf.gather(mu,     indices_mse, axis=1)
            )
            if pesos_mse is not None:
                w = tf.constant(pesos_mse, dtype=tf.float32)
                mse_vars = mse_vars * w
            loss_total += tf.reduce_mean(mse_vars)

        nll_base = -y_pred.log_prob(y_true)

        if dinamico:
            error_absoluto_global = tf.abs(y_true - mu)
            error_medio_muestra   = tf.reduce_mean(error_absoluto_global, axis=-1)
            nll_final = nll_base * (1.0 + factor_sensibilidad * error_medio_muestra)
        else:
            nll_final = nll_base

        return loss_total + (peso_nll * tf.reduce_mean(nll_final))

    return loss_personalizada

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=crear_loss_bayesiana(
        indices_mae=[3],              
        indices_mse=[0, 1, 2],        
        pesos_mae=[2.0],              
        pesos_mse=[1.5, 0.3, 1.5],   
        peso_nll=0.05,                
        dinamico=True,
        factor_sensibilidad=5.0       
    )
)

model.summary()

# MAE para el viento porque es mas aleatoria y tiene mas picos bruscos. El mse lo castiga demasiado y no aprende bien
# MSE para las otras porque son o mas periodicas o mas estables. 


# 5. ENTRENAMIENTO

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Desescalado 

def inverse_multi(scaler, data_to_inverse, n_features):
    dummy = np.zeros((len(data_to_inverse), n_features))
    for i in range(5):
        dummy[:, i] = data_to_inverse[:, i]
    rescaled = scaler.inverse_transform(dummy)
    return rescaled[:, 0], rescaled[:, 1], rescaled[:, 2], rescaled[:, 3], rescaled[:, 4]


# Muestreo de MC

N_INICIO = 0
N_PASOS = 96   
N_MC    = 100

print(f"\nINICIANDO PREDICCION ({N_PASOS} pasos, {N_MC} trayectorias MC)")


ventana_mc = np.tile(X_test[N_INICIO][np.newaxis, :, :], (N_MC, 1, 1))

pred_mean_global = np.zeros((N_PASOS, 5), dtype=np.float32)
pred_std_global  = np.zeros((N_PASOS, 5), dtype=np.float32)

for paso in range(N_PASOS):

    dist_paso       = model(ventana_mc, training=True)
    muestras_scaled = dist_paso.sample().numpy()  # (N_MC, 5)

    media_paso = muestras_scaled.mean(axis=0)
    pred_mean_global[paso] = media_paso
    pred_std_global[paso]  = muestras_scaled.std(axis=0)

    nueva_fila = np.tile(media_paso[np.newaxis, np.newaxis, :], (N_MC, 1, 1))
    ventana_mc = np.concatenate(
        [ventana_mc[:, 1:, :], nueva_fila], axis=1
    )

    if (paso + 1) % 24 == 0:
        print(f"  Paso {paso + 1}/{N_PASOS} completado")


# Desescalado

pred_temp,  pred_presion,  pred_humedad,  pred_viento,  pred_rad = inverse_multi(scaler, pred_mean_global, n_features)

upper_scaled = pred_mean_global + 1.96 * pred_std_global
lower_scaled = pred_mean_global - 1.96 * pred_std_global

upper_temp, upper_presion, upper_humedad, upper_viento, upper_rad = inverse_multi(scaler, upper_scaled, n_features)
lower_temp, lower_presion, lower_humedad, lower_viento, lower_rad = inverse_multi(scaler, lower_scaled, n_features)

std_temp    = upper_temp    - pred_temp
std_presion = upper_presion - pred_presion
std_humedad = upper_humedad - pred_humedad
std_viento  = upper_viento  - pred_viento
std_rad     = upper_rad     - pred_rad

real_temp, real_presion, real_humedad, real_viento, real_rad = inverse_multi(
    scaler, y_test[N_INICIO:N_INICIO + N_PASOS], n_features)



# 7. VISUALIZACION

limite1 = 0
limite2 = N_PASOS

fig, axs = plt.subplots(4, 1, figsize=(22, 12), sharex=True)

def plot_v(ax, real, pred, upper, lower, color, title, ylabel):
    r = real[limite1:limite2]
    p = pred[limite1:limite2]
    u = upper[limite1:limite2]
    l = lower[limite1:limite2]
    x = np.arange(len(p))
    ax.plot(x, r, color='black', alpha=0.5, label='Real', linewidth=1)
    ax.plot(x, p, color=color, linestyle='--', label='Predicción media', linewidth=1.5)
    ax.fill_between(x, l, u, color=color, alpha=0.2, label='IC 95%')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_v(axs[0], real_temp,    pred_temp,    upper_temp,    lower_temp,    'red',    'Temperatura (°C)',       '°C')
plot_v(axs[1], real_presion, pred_presion, upper_presion, lower_presion, 'orange', 'Presión (hPa)',          'hPa')
plot_v(axs[2], real_humedad, pred_humedad, upper_humedad, lower_humedad, 'green',  'Humedad (%)',            '%')
plot_v(axs[3], real_viento,  pred_viento,  upper_viento,  lower_viento,  'blue',   'Velocidad viento (m/s)', 'm/s')

plt.xlabel('Horas predichas')
plt.tight_layout()
plt.savefig("Hibrido24.png", dpi=300, bbox_inches='tight')
plt.show()



# 8. METRICAS


from scipy import stats


def calcular_picp(real, pred, std):
    dentro = np.logical_and(real >= pred - std, real <= pred + std)
    return np.mean(dentro) * 100

def calcular_mpiw(upper, lower):
    return np.mean(upper - lower)

def calcular_pinaw(upper, lower, real):
    rango = real.max() - real.min()
    return np.mean(upper - lower) / rango if rango > 0 else np.nan

def calcular_winkler(real, upper, lower, alpha=0.05):
    ancho = upper - lower
    pen   = np.where(real < lower, 2/alpha * (lower - real),
            np.where(real > upper, 2/alpha * (real - upper), 0.0))
    return np.mean(ancho + pen)

def calcular_skill(real, pred):
    baseline     = np.roll(real, 1)[1:]
    mse_modelo   = np.mean((real[1:] - pred[1:])**2)
    mse_baseline = np.mean((real[1:] - baseline)**2)
    return 1.0 - (mse_modelo / mse_baseline) if mse_baseline > 0 else np.nan

def calcular_mape(real, pred, umbral=0.5):
    mask = np.abs(real) > umbral
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100

def calcular_bias(real, pred):
    return np.mean(pred - real)

def calcular_mae_picos(real, pred, percentil=90):
    umbral_alto = np.percentile(real, percentil)
    umbral_bajo = np.percentile(real, 100 - percentil)
    mask_alto   = real >= umbral_alto
    mask_bajo   = real <= umbral_bajo
    mae_alto = mean_absolute_error(real[mask_alto], pred[mask_alto]) if mask_alto.sum() > 0 else np.nan
    mae_bajo = mean_absolute_error(real[mask_bajo], pred[mask_bajo]) if mask_bajo.sum() > 0 else np.nan
    return mae_alto, mae_bajo

def calcular_hit_rate(real, pred, umbral, tipo='mayor'):
    evento_real = (real > umbral) if tipo == 'mayor' else (real < umbral)
    evento_pred = (pred > umbral) if tipo == 'mayor' else (pred < umbral)
    if evento_real.sum() == 0:
        return np.nan, np.nan
    hit  = (evento_real & evento_pred).sum() / evento_real.sum() * 100
    if evento_pred.sum() == 0:
        falsa = np.nan
    else:
        falsa = (~evento_real & evento_pred).sum() / evento_pred.sum() * 100
    return hit, falsa

def calcular_ece(real_sc, mu_sc, std_sc):
    niveles = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    errores = []
    for alpha in niveles:
        z = stats.norm.ppf((1 + alpha) / 2)
        dentro = np.logical_and(
            real_sc >= mu_sc - z * std_sc,
            real_sc <= mu_sc + z * std_sc
        )
        errores.append(abs(np.mean(dentro) - alpha))
    return np.mean(errores) * 100  # en %


UMBRALES = {
    "TEMPERATURA": (5.0,  'menor'),   
    "PRESION":     (1010, 'menor'),   
    "HUMEDAD":     (85.0, 'mayor'),   
    "VIENTO":      (10.0, 'mayor'),   
    "RADIACIÓN":   (400,  'mayor'),   
}


pred_vars  = [pred_temp,  pred_presion,  pred_humedad,  pred_viento,  pred_rad]
real_vars  = [real_temp,  real_presion,  real_humedad,  real_viento,  real_rad]
upper_vars = [upper_temp, upper_presion, upper_humedad, upper_viento, upper_rad]
lower_vars = [lower_temp, lower_presion, lower_humedad, lower_viento, lower_rad]
std_vars   = [std_temp,   std_presion,   std_humedad,   std_viento,   std_rad]

nombres = ["TEMPERATURA", "PRESION", "HUMEDAD", "VIENTO", "RADIACIÓN"]


# TABLA 1 — Prediccion puntual

print(f"\n{'='*85}")
print("TABLA 1 — PREDICCIÓN PUNTUAL")
print(f"{'='*85}")
print(f"{'VARIABLE':<12} | {'MAE':<7} | {'RMSE':<7} | {'R²':<7} | {'Bias':<8} | {'MAPE %':<8} | {'Skill':<7} | {'MAE p90↑':<9} | {'MAE p10↓'}")
print("-" * 85)

for j, nombre in enumerate(nombres):
    real = real_vars[j]
    pred = pred_vars[j]

    mae   = mean_absolute_error(real, pred)
    rmse  = np.sqrt(mean_squared_error(real, pred))
    r2    = r2_score(real, pred)
    bias  = calcular_bias(real, pred)
    mape  = calcular_mape(real, pred)
    skill = calcular_skill(real, pred)
    mae_alto, mae_bajo = calcular_mae_picos(real, pred, percentil=90)

    mape_s  = f"{mape:.1f}"   if not np.isnan(mape)  else "N/A"
    skill_s = f"{skill:.3f}"  if not np.isnan(skill) else "N/A"
    malt_s  = f"{mae_alto:.3f}" if not np.isnan(mae_alto) else "N/A"
    mbaj_s  = f"{mae_bajo:.3f}" if not np.isnan(mae_bajo) else "N/A"

    print(f"{nombre:<12} | {mae:<7.3f} | {rmse:<7.3f} | {r2:<7.3f} | {bias:<8.3f} | {mape_s:<8} | {skill_s:<7} | {malt_s:<9} | {mbaj_s}")

print("-" * 85)


# TABLA 2 — Banda de incertidumbre

print(f"\n{'='*75}")
print("TABLA 2 — BANDA DE INCERTIDUMBRE")
print(f"{'='*75}")
print(f"{'VARIABLE':<12} | {'PICP %':<8} | {'PINAW':<8} | {'MPIW':<8} | {'Winkler':<10} | {'ECE %'}")
print("-" * 65)

for j, nombre in enumerate(nombres):
    real  = real_vars[j]
    pred  = pred_vars[j]
    std   = std_vars[j]
    upper = upper_vars[j]
    lower = lower_vars[j]

    real_sc = y_test[N_INICIO:N_INICIO+N_PASOS, j]
    mu_sc   = pred_mean_global[:, j]
    std_sc  = pred_std_global[:, j]

    picp    = calcular_picp(real, pred, std)
    pinaw   = calcular_pinaw(upper, lower, real)
    mpiw    = calcular_mpiw(upper, lower)
    winkler = calcular_winkler(real, upper, lower)
    ece     = calcular_ece(real_sc, mu_sc, std_sc)

    print(f"{nombre:<12} | {picp:<8.2f} | {pinaw:<8.3f} | {mpiw:<8.3f} | {winkler:<10.3f} | {ece:.2f}")

print("-" * 65)


# TABLA 3 — Hit Rate con umbral

print(f"\n{'='*65}")
print("TABLA 3 — DETECCION DE EVENTOS EXTREMOS (Hit Rate con umbral)")
print(f"{'='*65}")
print(f"{'VARIABLE':<12} | {'Umbral':<10} | {'Tipo':<7} | {'Hit Rate %':<12} | {'Falsa Alarma %'}")
print("-" * 65)

for j, nombre in enumerate(nombres):
    real = real_vars[j]
    pred = pred_vars[j]
    umbral, tipo = UMBRALES[nombre]

    hit, falsa = calcular_hit_rate(real, pred, umbral, tipo)

    hit_s   = f"{hit:.1f}"   if not np.isnan(hit)   else "N/A (sin eventos)"
    falsa_s = f"{falsa:.1f}" if not np.isnan(falsa) else "N/A"
    tipo_str = f">{umbral}" if tipo == 'mayor' else f"<{umbral}"

    print(f"{nombre:<12} | {tipo_str:<10} | {tipo:<7} | {hit_s:<12} | {falsa_s}")

print("-" * 65)


# TABLA 4 — Calibracion bayesiana 

print(f"\n{'='*80}")
print("TABLA 4 — CALIBRACIÓN BAYESIANA (PICP empírico vs nominal)")
print(f"{'='*80}")
print(f"{'Nivel nominal':<16}", end="")
for nombre in nombres:
    print(f" | {nombre:<12}", end="")
print()
print("-" * 80)

niveles = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
for alpha in niveles:
    z = stats.norm.ppf((1 + alpha) / 2)
    print(f"{alpha*100:.0f}%{'':<13}", end="")
    for j in range(len(nombres)):
        real_sc = y_test[N_INICIO:N_INICIO+N_PASOS, j]
        mu_sc   = pred_mean_global[:, j]
        std_sc  = pred_std_global[:, j]
        dentro  = np.logical_and(
            real_sc >= mu_sc - z * std_sc,
            real_sc <= mu_sc + z * std_sc
        )
        print(f" | {np.mean(dentro)*100:<12.1f}", end="")
    print()

print("-" * 80)





# 9. ANALISIS PRIMERAS 48 HORAS
=
print(f"\n{'='*85}")
print("INICIANDO ANÁLISIS DE CORTO ALCANCE (PRIMERAS 48 HORAS)")
print(f"{'='*85}")

H_ZOOM = min(48, N_PASOS)  
x_48 = np.arange(H_ZOOM)

fig_zoom, axs_zoom = plt.subplots(2, 2, figsize=(16, 10))

def plot_zoom_48h(ax, real, pred, upper, lower, color, title, ylabel):
    ax.plot(x_48, real[:H_ZOOM], color='black', alpha=0.7, label='Real', linewidth=1.5)
    ax.plot(x_48, pred[:H_ZOOM], color=color, linestyle='--', label='Predicción media', linewidth=2)
    ax.fill_between(x_48, lower[:H_ZOOM], upper[:H_ZOOM], color=color, alpha=0.2, label='IC 95%')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plot_zoom_48h(axs_zoom[0, 0], real_temp, pred_temp, upper_temp, lower_temp, 'red', 'Temperatura (Primeras 48h)', '°C')
plot_zoom_48h(axs_zoom[0, 1], real_presion, pred_presion, upper_presion, lower_presion, 'orange', 'Presión (Primeras 48h)', 'hPa')
plot_zoom_48h(axs_zoom[1, 0], real_humedad, pred_humedad, upper_humedad, lower_humedad, 'green', 'Humedad (Primeras 48h)', '%')
plot_zoom_48h(axs_zoom[1, 1], real_viento, pred_viento, upper_viento, lower_viento, 'blue', 'Viento (Primeras 48h)', 'm/s')

plt.tight_layout()
plt.savefig("Zoom_48h_Hibrido_2x2.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"{'VARIABLE':<12} | {'MAE (48h)':<9} | {'RMSE (48h)':<10} | {'R² (48h)':<9} | {'PICP %':<8} | {'PINAW':<8} | {'Winkler'}")
print("-" * 85)

for j, nombre in enumerate(["TEMPERATURA", "PRESION", "HUMEDAD", "VIENTO"]):
    r_48 = real_vars[j][:H_ZOOM]
    p_48 = pred_vars[j][:H_ZOOM]
    u_48 = upper_vars[j][:H_ZOOM]
    l_48 = lower_vars[j][:H_ZOOM]
    s_48 = std_vars[j][:H_ZOOM]

    mae_48  = mean_absolute_error(r_48, p_48)
    rmse_48 = np.sqrt(mean_squared_error(r_48, p_48))
    r2_48   = r2_score(r_48, p_48)

    picp_48    = calcular_picp(r_48, p_48, s_48)
    pinaw_48   = calcular_pinaw(u_48, l_48, r_48)
    winkler_48 = calcular_winkler(r_48, u_48, l_48)

    print(f"{nombre:<12} | {mae_48:<9.3f} | {rmse_48:<10.3f} | {r2_48:<9.3f} | {picp_48:<8.2f} | {pinaw_48:<8.3f} | {winkler_48:.3f}")

print("-" * 85)