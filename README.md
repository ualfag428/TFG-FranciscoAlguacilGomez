# TFG-FranciscoAlguacilGomez

# Análisis comparativo de arquitecturas híbridas bayesianas para predicción de series climáticas

Este repositorio contiene los experimentos desarrollados para un Trabajo de Fin de Grado sobre predicción probabilística de series climáticas mediante arquitecturas neuronales profundas. El modelo principal es [`HIBRIDO_24.py`](./HIBRIDO_24.py), una red híbrida que combina LSTM, Transformer y capas bayesianas `DenseFlipout` para predecir variables meteorológicas horarias con intervalos de incertidumbre.

El README está organizado en dos niveles:

1. Explicación detallada, bloque por bloque, del modelo `HIBRIDO_24.py`.
2. Comparación con los modelos restantes, explicand las partes que cambian en estructura o configuración.

## Índice

- [Objetivo del proyecto](#objetivo-del-proyecto)
- [Dataset y variables](#dataset-y-variables)
- [Modelo principal: HIBRIDO_24.py](#modelo-principal-hibrido_24py)
- [Predicción autoregresiva e incertidumbre](#predicción-autoregresiva-e-incertidumbre)
- [Métricas de evaluación](#métricas-de-evaluación)
- [Ejemplos de ejecución](#ejemplos-de-ejecución)
- [Comparación con otros modelos](#comparación-con-otros-modelos)
- [Consideraciones metodológicas](#consideraciones-metodológicas)
- [Observaciones de mantenimiento](#observaciones-de-mantenimiento)

## Objetivo del proyecto

El objetivo es comparar distintas arquitecturas bayesianas para la predicción de series climáticas multivariantes. Cada modelo recibe una ventana histórica de observaciones meteorológicas y predice el siguiente estado climático. A partir de esa predicción, el código genera trayectorias futuras de forma autoregresiva.

La aportación principal no es solo obtener una predicción puntual, sino estimar también la incertidumbre asociada. Para ello, la salida del modelo se formula como una distribución de probabilidad y se emplea muestreo Monte Carlo durante la inferencia.

En términos prácticos, el sistema permite responder preguntas como:

- Qué valor medio se espera para temperatura, presión, humedad, viento y radiación.
- Qué intervalo de confianza acompaña a cada predicción.
- Qué variables se predicen con mayor estabilidad.
- Qué arquitectura captura mejor los patrones temporales.
- Si una ventana de 24 horas o de 168 horas mejora la predicción.

## Dataset y variables

Los scripts utilizan el archivo:

```text
datos_climaticos07.txt
```

Se seleccionan cinco variables:

| Índice | Columna | Unidad | Descripción |
|---:|---|---|---|
| 0 | `temperatura_c` | °C | Temperatura del aire. |
| 1 | `presion_hpa` | hPa | Presión atmosférica. |
| 2 | `humedad_relativa` | % | Humedad relativa. |
| 3 | `velocidad_viento_10m` | m/s | Velocidad del viento a 10 metros. |
| 4 | `radiacion_ghi` | W/m2 | Radiación global horizontal. |

Todas las variables se usan como entrada y como objetivo. Por tanto, el modelo aprende una relación multivariante:

```text
[temperatura, presión, humedad, viento, radiación] pasado
        -> [temperatura, presión, humedad, viento, radiación] futuro
```

## Modelo principal: HIBRIDO_24.py

`HIBRIDO_24.py` es el modelo central del proyecto. Utiliza una ventana de 24 horas, por lo que cada muestra de entrada representa un día completo de condiciones meteorológicas.

### 1. Importación de librerías

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Cada grupo cumple una función concreta:

| Librería | Uso |
|---|---|
| `pandas` | Lectura del dataset tabular. |
| `numpy` | Operaciones numéricas, ventanas, arrays y métricas auxiliares. |
| `matplotlib` | Visualización de predicciones e intervalos. |
| `MinMaxScaler` | Normalización de variables climáticas. |
| `tensorflow.keras` | Construcción de la arquitectura neuronal. |
| `tensorflow_probability` | Capas bayesianas y distribuciones predictivas. |
| `sklearn.metrics` | Métricas de error puntual. |

La elección de TensorFlow Probability es clave: permite que la red produzca distribuciones y no solamente valores deterministas.

### 2. Semillas y reproducibilidad

```python
SEED = 50
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Este bloque fija semillas para reducir la variabilidad entre ejecuciones. Se controlan tres fuentes de aleatoriedad:

- `random`, usado por Python.
- `numpy`, usado en creación y manipulación de arrays.
- `tensorflow`, usado en inicialización de pesos, dropout y operaciones internas.

Esto mejora la reproducibilidad, aunque no garantiza determinismo absoluto si se usa GPU o ciertas operaciones internas no deterministas.

### 3. Alias de distribuciones

```python
tfd = tfp.distributions
```

Se crea un alias para acceder de forma más limpia a distribuciones probabilísticas de TensorFlow Probability. Más adelante se usa para definir una distribución normal multivariante independiente:

```python
tfd.Normal(...)
tfd.Independent(...)
```

### 4. Carga de datos

```python
df = pd.read_csv(r'C:\Users\pacoa\Documents\Universidad-Grado en Matematicas\CUARTO CURSO\TFG\Python\datos_climaticos07.txt')
```

El script carga el dataset desde una ruta absoluta. Este enfoque funciona en el equipo local, pero para hacer el proyecto más portable sería recomendable usar rutas relativas o un parámetro de configuración.

Después se seleccionan las columnas:

```python
features = [
    'temperatura_c',
    'presion_hpa',
    'humedad_relativa',
    'velocidad_viento_10m',
    'radiacion_ghi'
]

n_features = len(features)
data = df[features].values.astype(np.float32)
```

La conversión a `np.float32` es adecuada para TensorFlow: reduce memoria y evita conversiones implícitas durante el entrenamiento.

### 5. Definición de ventana e horizonte

```python
window_size = 24
horizon = 1
```

`window_size = 24` indica que cada entrada contiene 24 horas consecutivas. `horizon = 1` indica que el objetivo inmediato es la hora siguiente.

Con cinco variables, cada muestra de entrada tiene forma:

```text
(24, 5)
```

y cada salida tiene forma:

```text
(5,)
```

Es decir:

```text
24 horas pasadas x 5 variables -> 1 hora futura x 5 variables
```

### 6. División temporal entrenamiento/test

```python
total_windows = len(data) - window_size - horizon + 1
test_size  = int(total_windows * 0.2)
train_size = total_windows - test_size
```

El número de ventanas depende de la longitud del dataset. El script reserva aproximadamente el 80% inicial para entrenamiento y el 20% final para test.

Esto es correcto para series temporales, porque respeta el orden cronológico. No se debe mezclar aleatoriamente entrenamiento y test en este tipo de problema, ya que se introduciría información futura en el entrenamiento.

### 7. Escalado sin fuga de información

```python
train_cut_index = train_size + window_size - 1
scaler = MinMaxScaler()
scaler.fit(data[:train_cut_index])
scaled_data = scaler.transform(data)
```

Este bloque es metodológicamente importante. El `MinMaxScaler` se ajusta solo con datos correspondientes al tramo de entrenamiento. Después se transforma el dataset completo con esos parámetros.

La idea es evitar `data leakage`: si el escalador se ajustara con todo el dataset, los mínimos y máximos del test influirían en el entrenamiento. Aunque parezca un detalle menor, en series temporales puede contaminar la evaluación.

`MinMaxScaler` transforma cada variable a un rango comparable, habitualmente `[0, 1]`. Esto ayuda a que la red entrene de forma estable, ya que temperatura, presión, humedad, viento y radiación tienen escalas físicas muy distintas.

### 8. Creación de ventanas multivariantes

```python
def create_windows_multi(dataset, window_size=24, horizon=1, salto=1):
    X, y = [], []
    for i in range(0, len(dataset) - window_size - horizon + 1, salto):
        X.append(dataset[i:(i + window_size), :])
        target_idx = i + window_size + horizon - 1
        y.append(dataset[target_idx, 0:5])
    return np.array(X), np.array(y)
```

Esta función convierte la serie temporal continua en un problema supervisado.

Para cada índice `i`:

- `X` recibe las 24 horas desde `i` hasta `i + 23`.
- `y` recibe la observación futura situada en `i + 24`.

La variable `salto=1` indica que se crea una ventana desplazándose hora a hora. Esto maximiza el número de ejemplos de entrenamiento.

Ejemplo conceptual:

```text
X[0] = horas 0  ... 23    -> y[0] = hora 24
X[1] = horas 1  ... 24    -> y[1] = hora 25
X[2] = horas 2  ... 25    -> y[2] = hora 26
```

Después se separan entrenamiento y test:

```python
X, y = create_windows_multi(scaled_data, window_size, horizon, salto=1)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```

### 9. Codificación posicional

El Transformer necesita información de posición. La atención por sí sola compara elementos de una secuencia, pero no sabe si una observación corresponde a la primera, décima o última hora de la ventana.

```python
class PositionalEncoding(layers.Layer):
    def __init__(self, max_steps, max_dims, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        if max_dims % 2 == 1: max_dims += 1
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2]  = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_encoding = tf.constant(pos_emb, dtype=tf.float32)
```

Este bloque construye una matriz de codificación sinusoidal. Las posiciones pares usan seno y las impares coseno. Es una técnica clásica de los Transformers.

El método `call` suma esa codificación a la entrada:

```python
def call(self, inputs):
    shape = tf.shape(inputs)
    return inputs + self.positional_encoding[:, :shape[1], :shape[2]]
```

Si la entrada tiene forma:

```text
(batch, 24, 128)
```

la codificación posicional también se adapta a:

```text
(1, 24, 128)
```

y se suma por broadcasting.

### 10. Entrada del modelo

```python
input_layer = layers.Input(shape=(window_size, n_features))
```

Para `HIBRIDO_24.py`, esto equivale a:

```text
Input(shape=(24, 5))
```

Cada muestra contiene 24 horas y cinco variables.

### 11. Rama LSTM

```python
lstm_branch = layers.LSTM(64, return_sequences=False, dropout=0.2)(input_layer)
```

La rama LSTM transforma la secuencia completa en un vector de 64 componentes. Como `return_sequences=False`, la capa devuelve solo el último estado oculto.

Su función es capturar dinámica secuencial:

- inercia térmica,
- transiciones suaves de presión,
- evolución reciente de humedad,
- cambios acumulados en viento,
- patrones horarios dentro de las últimas 24 horas.

El `dropout=0.2` ayuda a regularizar la red, reduciendo sobreajuste.

### 12. Rama Transformer

La segunda rama procesa la misma entrada con atención.

Primero se proyectan las cinco variables a un espacio de 128 dimensiones:

```python
trans_branch = layers.Dense(128)(input_layer)
```

La forma pasa de:

```text
(batch, 24, 5)
```

a:

```text
(batch, 24, 128)
```

Después se añade codificación posicional:

```python
trans_branch = PositionalEncoding(max_steps=window_size, max_dims=128)(trans_branch)
```

Luego se normaliza:

```python
trans_branch = layers.LayerNormalization(epsilon=1e-6)(trans_branch)
```

La normalización estabiliza el entrenamiento y evita que las activaciones crezcan de forma descontrolada.

La atención multi-cabeza se aplica así:

```python
attention_out = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=32,
    dropout=0.1
)(trans_branch, trans_branch)
```

La consulta, clave y valor son la misma secuencia, por lo que se trata de auto-atención. Cada hora puede atender a cualquier otra hora de la ventana.

`num_heads=4` divide la atención en cuatro subespacios. `key_dim=32` implica que cada cabeza trabaja con vectores de dimensión 32. En conjunto, la atención puede modelar relaciones diferentes en paralelo.

Después se añade una conexión residual:

```python
trans_branch = layers.Add()([trans_branch, attention_out])
```

La conexión residual conserva la representación previa y suma la información aprendida por la atención. Esto ayuda a entrenar redes más estables.

La secuencia se comprime con:

```python
trans_branch = layers.GlobalMaxPooling1D()(trans_branch)
```

Este pooling toma el máximo a lo largo del eje temporal. Convierte:

```text
(batch, 24, 128)
```

en:

```text
(batch, 128)
```

Finalmente:

```python
trans_branch = layers.Dense(128, activation="relu")(trans_branch)
```

produce una representación densa final de la rama Transformer.

### 13. Fusión híbrida

```python
combined = layers.Concatenate()([lstm_branch, trans_branch])
```

La salida LSTM tiene dimensión 64 y la salida Transformer dimensión 128. Al concatenarlas se obtiene:

```text
64 + 128 = 192
```

Esta es la parte esencial del modelo híbrido. La red combina:

- memoria recurrente, útil para continuidad temporal;
- atención, útil para relaciones globales dentro de la ventana;
- aprendizaje bayesiano, útil para incertidumbre.

### 14. Factor KL para capas bayesianas

```python
N_train = len(X_train)
kl_factor = 1.0 / N_train
kl_divergence_function = lambda q, p, _: kl_factor * tfd.kl_divergence(q, p)
```

Las capas bayesianas añaden una penalización KL entre la distribución posterior aproximada de los pesos y la distribución previa. Dividir por `N_train` normaliza esa penalización respecto al tamaño del conjunto de entrenamiento.

Este término evita que las distribuciones de pesos se alejen sin control del prior y actúa como regularización bayesiana.

### 15. Bloque denso y capas Flipout

```python
combined = layers.Dense(128, activation='relu')(combined)
combined = layers.Dropout(0.25)(combined)
combined = tfp.layers.DenseFlipout(64, activation='relu', kernel_divergence_fn=kl_divergence_function)(combined)
combined = tfp.layers.DenseFlipout(32, activation='relu', kernel_divergence_fn=kl_divergence_function)(combined)
```

Primero se aplica una capa densa determinista de 128 neuronas. Después se añade `Dropout(0.25)`.

Las dos capas `DenseFlipout` son la parte bayesiana del modelo. A diferencia de una `Dense` convencional, no aprenden un peso fijo, sino una distribución sobre los pesos. Durante el entrenamiento e inferencia se muestrean perturbaciones eficientes de esos pesos.

La técnica Flipout reduce la varianza del estimador de gradiente frente a otros métodos de muestreo bayesiano, lo que permite entrenar redes probabilísticas de forma más estable.

### 16. Parámetros de la distribución de salida

```python
params = layers.Dense(tfp.layers.IndependentNormal.params_size(5))(combined)
```

La salida no son directamente cinco valores. Se generan los parámetros necesarios para una distribución normal independiente de dimensión 5.

Para cinco variables, se necesitan:

- 5 medias,
- 5 escalas o desviaciones típicas.

Por tanto, esta capa produce 10 parámetros.

### 17. Capa probabilística de salida

```python
output_layer = tfp.layers.DistributionLambda(
    lambda t: tfd.Independent(
        tfd.Normal(
            loc=t[..., :5],
            scale=0.02 + tf.math.softplus(t[..., 5:] * 0.3)
        ),
        reinterpreted_batch_ndims=1
    )
)(params)
```

Este bloque convierte los parámetros en una distribución real.

`loc=t[..., :5]` toma las primeras cinco salidas como medias predictivas.

`scale=0.02 + softplus(...)` toma las cinco restantes como escala. La función `softplus` garantiza que la desviación típica sea positiva. El término `0.02` impone un mínimo de incertidumbre para evitar escalas cercanas a cero.

`IndependentNormal` interpreta las cinco normales como una distribución conjunta independiente:

```text
P(temp, presión, humedad, viento, radiación)
```

con una media y una desviación típica por variable.

Finalmente se crea el modelo:

```python
model = models.Model(inputs=input_layer, outputs=output_layer)
```

### 18. Función de pérdida personalizada

La función:

```python
crear_loss_bayesiana(...)
```

combina error puntual y entrenamiento probabilístico.

Dentro de la pérdida:

```python
mu = y_pred.mean()
```

se extrae la media de la distribución predictiva. Esa media se usa para calcular MAE y MSE.

#### MAE para viento

```python
indices_mae=[3]
pesos_mae=[2.0]
```

El índice 3 corresponde a `velocidad_viento_10m`. Se usa MAE porque el viento es una variable irregular, con picos y cambios bruscos. El MSE penalizaría demasiado los errores grandes y podría provocar un aprendizaje menos robusto.

#### MSE para temperatura, presión y humedad

```python
indices_mse=[0, 1, 2]
pesos_mse=[1.5, 0.3, 1.5]
```

Temperatura, presión y humedad se tratan con MSE. Estas variables suelen tener evolución más suave o estructurada. Los pesos ajustan la importancia relativa:

- temperatura: peso 1.5,
- presión: peso 0.3,
- humedad: peso 1.5.

La presión recibe menos peso porque su escala física y su comportamiento pueden dominar la pérdida si se penaliza demasiado.

#### Negative Log-Likelihood

```python
nll_base = -y_pred.log_prob(y_true)
```

La NLL entrena la distribución completa. No basta con que la media sea buena: el modelo también debe asignar probabilidad alta al valor real.

Después se aplica ponderación dinámica:

```python
error_absoluto_global = tf.abs(y_true - mu)
error_medio_muestra = tf.reduce_mean(error_absoluto_global, axis=-1)
nll_final = nll_base * (1.0 + factor_sensibilidad * error_medio_muestra)
```

Cuando la media se equivoca más, la NLL pesa más. Esto fuerza al modelo a corregir tanto la predicción central como su incertidumbre.

La pérdida final es:

```python
return loss_total + (peso_nll * tf.reduce_mean(nll_final))
```

### 19. Compilación

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=crear_loss_bayesiana(...)
)
```

Se usa Adam con `learning_rate=0.0005`. Es una tasa moderada, adecuada para redes con componentes recurrentes, atención y capas probabilísticas.

### 20. Entrenamiento

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
]
```

`EarlyStopping` detiene el entrenamiento si la pérdida de validación no mejora durante 20 épocas. Además, restaura los mejores pesos.

`ReduceLROnPlateau` reduce el learning rate a la mitad si la validación se estanca durante 8 épocas.

El ajuste se lanza con:

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
```

El modelo entrena hasta 100 épocas con lotes de 128 muestras.

## Predicción autoregresiva e incertidumbre

### 1. Desescalado

```python
def inverse_multi(scaler, data_to_inverse, n_features):
    dummy = np.zeros((len(data_to_inverse), n_features))
    for i in range(5):
        dummy[:, i] = data_to_inverse[:, i]
    rescaled = scaler.inverse_transform(dummy)
    return rescaled[:, 0], rescaled[:, 1], rescaled[:, 2], rescaled[:, 3], rescaled[:, 4]
```

El modelo trabaja en espacio escalado. Esta función reconstruye arrays con cinco columnas para aplicar `inverse_transform` y volver a unidades físicas.

Devuelve cada variable por separado:

```text
temperatura, presión, humedad, viento, radiación
```

### 2. Parámetros de predicción

```python
N_INICIO = 0
N_PASOS = 96
N_MC = 100
```

`N_PASOS = 96` equivale a 96 horas futuras, es decir, cuatro días. `N_MC = 100` genera 100 trayectorias Monte Carlo para estimar incertidumbre.

### 3. Inicialización de trayectorias Monte Carlo

```python
ventana_mc = np.tile(X_test[N_INICIO][np.newaxis, :, :], (N_MC, 1, 1))
```

Se toma una ventana inicial del test y se replica 100 veces. Cada réplica seguirá una trayectoria probabilística distinta, aunque todas parten del mismo contexto histórico.

La forma resultante es:

```text
(100, 24, 5)
```

### 4. Bucle autoregresivo

```python
for paso in range(N_PASOS):
    dist_paso = model(ventana_mc, training=True)
    muestras_scaled = dist_paso.sample().numpy()
```

En cada paso, el modelo devuelve una distribución. Al llamar a `.sample()`, se genera una muestra para cada una de las 100 trayectorias.

El uso de `training=True` mantiene activos los mecanismos estocásticos del modelo, incluyendo las capas bayesianas y dropout. Esto es intencionado: se busca estimar incertidumbre predictiva.

Después se calcula:

```python
media_paso = muestras_scaled.mean(axis=0)
pred_mean_global[paso] = media_paso
pred_std_global[paso] = muestras_scaled.std(axis=0)
```

La media representa la predicción central. La desviación típica representa la dispersión entre trayectorias.

### 5. Actualización autoregresiva de la ventana

```python
nueva_fila = np.tile(media_paso[np.newaxis, np.newaxis, :], (N_MC, 1, 1))
ventana_mc = np.concatenate([ventana_mc[:, 1:, :], nueva_fila], axis=1)
```

Aquí está el núcleo de la predicción multi-step. La ventana pierde la hora más antigua y añade la predicción media como nueva hora final.

Esto simula una predicción real a futuro: después de predecir la primera hora, ya no se dispone del dato real siguiente, así que el modelo se alimenta de sus propias predicciones.

Consecuencia: el error puede acumularse a medida que aumenta el horizonte.

### 6. Intervalos al 95%

```python
upper_scaled = pred_mean_global + 1.96 * pred_std_global
lower_scaled = pred_mean_global - 1.96 * pred_std_global
```

Se construyen intervalos aproximados al 95% bajo hipótesis normal:

```text
media ± 1.96 desviaciones típicas
```

Después se desescalan junto con la predicción media y los valores reales.

### 7. Visualización

La función `plot_v` representa:

- serie real,
- predicción media,
- intervalo de confianza.

```python
ax.plot(x, r, color='black', alpha=0.5, label='Real')
ax.plot(x, p, color=color, linestyle='--', label='Predicción media')
ax.fill_between(x, l, u, color=color, alpha=0.2, label='IC 95%')
```

La figura principal se guarda como:

```text
Hibrido24.png
```

El análisis adicional de corto alcance genera:

```text
Zoom_48h_Hibrido_2x2.png
```

## Métricas de evaluación

El proyecto evalúa el rendimiento desde tres perspectivas: precisión puntual, calidad de incertidumbre y detección de eventos extremos.

### 1. Métricas puntuales

| Métrica | Interpretación |
|---|---|
| MAE | Error absoluto medio. Fácil de interpretar en unidades físicas. |
| RMSE | Penaliza errores grandes más que MAE. |
| R2 | Varianza explicada por el modelo. Puede ser negativo si el modelo es peor que una referencia media. |
| Bias | Media de `predicción - real`. Positivo indica sobreestimación; negativo, subestimación. |
| MAPE | Error porcentual absoluto medio. Se evita dividir por valores demasiado cercanos a cero. |
| Skill | Comparación frente a persistencia. Valores positivos indican mejora. |
| MAE p90/p10 | Error en extremos altos y bajos. |

### 2. Métricas probabilísticas

| Métrica | Interpretación |
|---|---|
| PICP | Porcentaje de valores reales dentro del intervalo. Para IC 95%, lo ideal es aproximarse a 95%. |
| MPIW | Anchura media del intervalo. |
| PINAW | Anchura normalizada por el rango real. |
| Winkler | Penaliza intervalos anchos y observaciones fuera del intervalo. Menor es mejor. |
| ECE | Error de calibración entre cobertura empírica y nominal. Menor es mejor. |

### 3. Eventos extremos

```python
UMBRALES = {
    "TEMPERATURA": (5.0,  'menor'),
    "PRESION":     (1010, 'menor'),
    "HUMEDAD":     (85.0, 'mayor'),
    "VIENTO":      (10.0, 'mayor'),
    "RADIACIÓN":   (400,  'mayor'),
}
```

Para cada variable se calcula:

- `Hit Rate`: porcentaje de eventos reales detectados.
- `Falsa Alarma`: porcentaje de alertas predichas sin evento real.

Estas métricas son útiles si el modelo se usa como sistema de alerta meteorológica.

## Ejemplos de ejecución

### Instalación de dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib scikit-learn scipy tensorflow tensorflow-probability
```

### Ejecutar el modelo híbrido 24h

```powershell
cd "C:\Users\pacoa\Documents\Universidad-Grado en Matematicas\CUARTO CURSO\TFG\Python"
python HIBRIDO_24.py
```

Configuración principal:

```python
window_size = 24
N_PASOS = 96
N_MC = 100
epochs = 100
```

Figura de ejemplo:

![Predicción primeras 48 horas - Híbrido 24h](../Imagenes%20Definitivas/Zoom481st24/zoom481st24.png)

Resultados de ejemplo extraídos de `zoom481st24-Hibrido.txt`:

| Variable | MAE 48h | RMSE 48h | R2 48h | PICP % | PINAW | Winkler |
|---|---:|---:|---:|---:|---:|---:|
| Temperatura | 1.294 | 1.500 | 0.770 | 97.92 | 0.457 | 5.319 |
| Presión | 2.435 | 3.179 | -7.114 | 60.42 | 1.094 | 45.346 |
| Humedad | 14.505 | 17.193 | 0.163 | 56.25 | 0.429 | 185.152 |
| Viento | 2.043 | 2.552 | 0.263 | 81.25 | 0.657 | 14.657 |

### Ejecutar el modelo híbrido 168h

```powershell
python HIBRIDO_168.py
```

Configuración principal:

```python
window_size = 168
N_PASOS = 200
N_MC = 100
epochs = 30
```

Figura de ejemplo:

![Predicción primeras 48 horas - Híbrido 168h](../Imagenes%20Definitivas/Zoom481st168/zoom481st168%20%28100%29%20200h%2030ep.png)

Esta variante permite estudiar si una semana de contexto histórico mejora la predicción frente a una ventana de 24 horas.

## Comparación con otros modelos

El proyecto incluye modelos LSTM puros, Transformer puros e híbridos. Todos comparten gran parte del flujo:

- carga de datos,
- escalado sin fuga de información,
- creación de ventanas,
- salida probabilística,
- pérdida bayesiana,
- predicción autoregresiva,
- métricas.

Lo que cambia principalmente es la arquitectura que extrae características temporales.

### 1. LSTM_24.py

Archivo: [`LSTM_24.py`](./LSTM_24.py)

El modelo LSTM puro elimina la rama Transformer. En lugar de combinar atención y recurrencia, usa dos capas LSTM apiladas:

```python
input_layer = layers.Input(shape=(window_size, n_features))

x = layers.LSTM(192, return_sequences=True, dropout=0.2)(input_layer)
x = layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
x = layers.LayerNormalization()(x)
```

Diferencias estructurales:

| Elemento | Híbrido 24h | LSTM 24h |
|---|---|---|
| Rama LSTM | Una LSTM de 64 unidades | Dos LSTM: 192 y 64 unidades |
| Rama Transformer | Sí | No |
| Codificación posicional | Sí | No |
| Atención multi-cabeza | Sí | No |
| Fusión de ramas | Concatenación LSTM + Transformer | No existe fusión |
| Bloque bayesiano | Igual | Igual |
| Salida probabilística | Igual | Igual |

La primera LSTM tiene `return_sequences=True`, por lo que devuelve una secuencia completa. Esa secuencia alimenta una segunda LSTM que resume la información temporal. Esto permite una extracción recurrente más profunda que en la rama LSTM del híbrido.

El resto del modelo se mantiene muy parecido:

```python
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.25)(x)
x = tfp.layers.DenseFlipout(64, activation='relu', kernel_divergence_fn=kl_divergence_function)(x)
x = tfp.layers.DenseFlipout(32, activation='relu', kernel_divergence_fn=kl_divergence_function)(x)
```

Por tanto, la comparación LSTM vs híbrido mide si añadir atención Transformer aporta información útil frente a una arquitectura recurrente más profunda.

### 2. TRANSFORMER_24.py

Archivo: [`TRANSFORMER_24.py`](./TRANSFORMER_24.py)

El modelo Transformer puro elimina la rama LSTM. Toda la extracción temporal se hace mediante atención.

```python
input_layer = layers.Input(shape=(window_size, n_features))

x = layers.Dense(192)(input_layer)
x = PositionalEncoding(max_steps=window_size, max_dims=192)(x)
x = layers.LayerNormalization(epsilon=1e-6)(x)
```

La proyección inicial sube de 5 variables a 192 dimensiones, más que en el híbrido, para compensar la ausencia de LSTM.

Después aplica dos bloques de atención:

```python
attn1 = layers.MultiHeadAttention(num_heads=4, key_dim=48, dropout=0.1)(x, x)
x = layers.Add()([x, attn1])
x = layers.LayerNormalization(epsilon=1e-6)(x)

attn2 = layers.MultiHeadAttention(num_heads=4, key_dim=48, dropout=0.1)(x, x)
x = layers.Add()([x, attn2])
x = layers.LayerNormalization(epsilon=1e-6)(x)
```

Diferencias estructurales:

| Elemento | Híbrido 24h | Transformer 24h |
|---|---|---|
| Rama LSTM | Sí | No |
| Dimensión latente Transformer | 128 | 192 |
| Bloques de atención | 1 | 2 |
| `key_dim` | 32 | 48 |
| Pooling temporal | `GlobalMaxPooling1D` | `GlobalMaxPooling1D` |
| Bloque bayesiano | Igual | Igual |
| Salida probabilística | Igual | Igual |

Después del segundo bloque de atención:

```python
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(192, activation="relu")(x)
```

Este modelo sirve para comprobar si la atención pura, sin recurrencia, es suficiente para modelar la evolución climática de corto plazo.

### 3. HIBRIDO_168.py

Archivo: [`HIBRIDO_168.py`](./HIBRIDO_168.py)

Esta variante conserva la misma arquitectura híbrida que `HIBRIDO_24.py`, pero cambia la longitud de la ventana:

```python
window_size = 168
```

Esto significa que cada muestra contiene una semana completa de observaciones horarias:

```text
168 horas x 5 variables
```

La arquitectura sigue siendo:

```text
Entrada -> LSTM(64)
        -> Dense(128) + PositionalEncoding + MultiHeadAttention
        -> Concatenate
        -> Dense + Dropout + DenseFlipout + DenseFlipout
        -> distribución normal independiente de 5 variables
```

Diferencias respecto a `HIBRIDO_24.py`:

| Parámetro | HIBRIDO_24.py | HIBRIDO_168.py |
|---|---:|---:|
| Semilla | 50 | 100 |
| Ventana | 24 h | 168 h |
| Épocas máximas | 100 | 30 |
| EarlyStopping patience | 20 | 8 |
| ReduceLROnPlateau patience | 8 | 4 |
| Pasos autoregresivos | 96 | 200 |
| Figura principal | `Hibrido24.png` | `Prueba_Simple_MC_168_v2C.png` |
| Figura 48h | `Zoom_48h_Hibrido_2x2.png` | `Primeras_48h_168h.png` |

La motivación científica es clara: con 168 horas el modelo puede observar ciclos más largos, especialmente patrones semanales o persistencias meteorológicas que no caben en una ventana diaria.

El coste es que la entrada es mucho más larga. Esto puede dificultar el entrenamiento y aumentar la acumulación de error en predicción autoregresiva.

### 4. LSTM_168.py

Archivo: [`LSTM_168.py`](./LSTM_168.py)

Es la versión semanal del modelo LSTM puro. Cambia la ventana, pero mantiene la estructura recurrente:

```python
window_size = 168

x = layers.LSTM(192, return_sequences=True, dropout=0.2)(input_layer)
x = layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
x = layers.LayerNormalization()(x)
```

Diferencias respecto a `LSTM_24.py`:

| Parámetro | LSTM_24.py | LSTM_168.py |
|---|---:|---:|
| Semilla | 50 | 100 |
| Ventana | 24 h | 168 h |
| Épocas máximas | 100 | 30 |
| Pasos autoregresivos | 96 | 200 |
| Figura | `LSTM_puro_24h.png` | `LSTM_puro_168h.png` |

La arquitectura recurrente no cambia. Lo que cambia es la cantidad de contexto temporal disponible para las LSTM.

### 5. TRANSFORMER_168.py

Archivo: [`TRANSFORMER_168.py`](./TRANSFORMER_168.py)

Es la versión semanal del Transformer puro. Conserva la misma idea que `TRANSFORMER_24.py`:

```python
x = layers.Dense(192)(input_layer)
x = PositionalEncoding(max_steps=window_size, max_dims=192)(x)
x = layers.LayerNormalization(epsilon=1e-6)(x)

attn1 = layers.MultiHeadAttention(num_heads=4, key_dim=48, dropout=0.1)(x, x)
x = layers.Add()([x, attn1])
x = layers.LayerNormalization(epsilon=1e-6)(x)

attn2 = layers.MultiHeadAttention(num_heads=4, key_dim=48, dropout=0.1)(x, x)
x = layers.Add()([x, attn2])
x = layers.LayerNormalization(epsilon=1e-6)(x)
```

Diferencias respecto a `TRANSFORMER_24.py`:

| Parámetro | TRANSFORMER_24.py | TRANSFORMER_168.py |
|---|---:|---:|
| Semilla | 50 | 100 |
| Ventana | 24 h | 168 h |
| Épocas máximas | 100 | 30 |
| Pasos autoregresivos | 96 | 200 |
| Dimensión latente | 192 | 192 |
| Bloques de atención | 2 | 2 |

En el Transformer, ampliar la ventana tiene una implicación importante: la atención compara posiciones entre sí. Pasar de 24 a 168 pasos incrementa considerablemente el número de relaciones temporales que el modelo puede analizar.

Observación técnica: en el archivo `TRANSFORMER_168.py` aparece una línea suelta con `=` antes de la sección de codificación posicional. Esa línea provoca un `SyntaxError` y debe eliminarse antes de ejecutar el script.

## Resumen comparativo de arquitecturas

| Modelo | Memoria recurrente | Atención | Ventana | Bloque bayesiano | Salida probabilística |
|---|---|---|---:|---|---|
| `LSTM_24.py` | Sí, 2 LSTM | No | 24 h | Sí | Sí |
| `TRANSFORMER_24.py` | No | Sí, 2 bloques | 24 h | Sí | Sí |
| `HIBRIDO_24.py` | Sí, 1 LSTM | Sí, 1 bloque | 24 h | Sí | Sí |
| `LSTM_168.py` | Sí, 2 LSTM | No | 168 h | Sí | Sí |
| `TRANSFORMER_168.py` | No | Sí, 2 bloques | 168 h | Sí | Sí |
| `HIBRIDO_168.py` | Sí, 1 LSTM | Sí, 1 bloque | 168 h | Sí | Sí |

## Consideraciones metodológicas

- La división temporal evita mezclar pasado y futuro.
- El escalado se ajusta solo con entrenamiento para evitar fuga de información.
- La salida probabilística permite evaluar calibración, no solo error medio.
- La predicción autoregresiva es más realista que predecir cada punto usando ventanas reales, pero acumula error.
- El uso de `validation_data=(X_test, y_test)` implica que el test actúa también como validación. Para una comparación final más estricta, convendría crear tres bloques: entrenamiento, validación y test.
- La radiación se predice y se evalúa, aunque varias figuras se centran solo en temperatura, presión, humedad y viento.


## Reproducibilidad

Los experimentos fijan semillas:

```python
SEED = 50   # Modelos 24h
SEED = 100  # Modelos 168h
```

Las ejecuciones con TensorFlow pueden variar ligeramente entre CPU y GPU. Para una comparación experimental definitiva, conviene ejecutar todos los modelos en el mismo entorno y registrar:

- versión de Python,
- versión de TensorFlow,
- versión de TensorFlow Probability,
- dispositivo usado,
- semilla,
- número de épocas reales entrenadas,
- fecha de ejecución.
