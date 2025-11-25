### Text2Market - Cecilia Podesta y M. Sol Vidal

Proyecto para generar señales de noticias financieras y usarlas junto con datos de mercado en un pipeline LSTM que predice el próximo valor de Adj Close a nivel diario por ticker. Incluye notebooks de lectura/limpieza de noticias, extracción de sentimiento con FinBERT, una recomendación con un LLM local, y un flujo de experimentos reproducibles.


## Estructura del repositorio


- `scores_news/`: notebooks de procesamiento de noticias.
  - `read_data_news.ipynb`: del dataset proveniente del paper de FINSPID (descargado de https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/Stock_news/nasdaq_exteral_data.csv) y análisis y filtrado inicial de la información.
  - `eda_y_seleccion_sub_dataset.ipynb`: 
    -Lectura del subdataset obtenido en el notebbok anterior y análisis de cantidad de noticias por ticker y en el tiempo.
    - Selección del subset de tickers y período de tiempo para trabajar en los siguientes steps del proyecto.
  - `deteccion_tickers.ipynb`: pipeline para la detección de tickers presentes de forma explícita en las noticias a partir de un patrón Regex.
  - `deteccion_sent_financiero_con_finbert.ipynb`: scoring de sentimiento con `ProsusAI/finbert` para las noticias con 512 tokens o menos.
  - `recomendacion_ollama.ipynb`: recomendación (1–5) con un modelo de Ollama (modelo: martain7r/finance-llama-8b:fp16) mediante un prompt que devuelve un valor de 1 a 5 para cada ticker/notica donde:
```
1 = Strongly Negative (Do not invest)
3 = Neutral
5 = Strongly Positive (Strong Buy)
```
- `clean_new_model/`:
  - `dataset_para_lstm.csv`: insumo para construir el dataset final.
  - `dataset_ceci_sol/`: CSVs finales por ticker con features de precio + noticias (ej. `AAPL.csv`, `MSFT.csv`, …).
  - `dfs_por_ticker_csv/`: salidas de resultados por ticker exportadas desde los experimentos.
  - `experiments_clean.ipynb`: notebook de experimentos; orquesta el pipeline y usa `lstm_pipeline.py`.
  - `sentiment_analysis.ipynb`: construye el dataset final de scores de noticias + datos financieros.
  - `lstm_pipeline.py`: implementación del modelo y utilidades de entrenamiento/evaluación en PyTorch.


## Flujo de datos (alto nivel)

1) Noticias externas (FNSPID en HuggingFace) → lectura/limpieza/selección (`scores_news/*`).
2) Enriquecimiento con señales:
   - Sentimiento financiero con `ProsusAI/finbert` (pos/neu/neg).
   - Recomendación LLM (valor 1–5) con Ollama.
3) Unión con datos de mercado y construcción del dataset final para LSTM (`sentiment_analysis.ipynb` → `dataset_ceci_sol/*`).
4) Entrenamiento y evaluación del LSTM (`experiments_clean.ipynb` usando `lstm_pipeline.py`).
5) Exportación de resultados por ticker (`dfs_por_ticker_csv/*`) y reporte (`Text2Market - Informe (1).pdf`).


## Dataset esperado para el LSTM

Cada CSV por ticker en `clean_new_model/dataset_ceci_sol/` contiene columnas típicas de OHLCV y señales derivadas de noticias. Ejemplo de encabezados (AAPL):

- Mercado: `Date`, `Adj Close`, `Close`, `High`, `Low`, `Open`, `Volume`, `VIX`
- Sentimiento FinBERT: `puntaje_sent_fin_pos`, `puntaje_sent_fin_neu`, `puntaje_sent_fin_neg` (+`has_news`)
- Señal Ollama: `recom_inv_llm` (+`has_news`)
- Score ChatGPT del paper FNSPID: `Sentiment_gpt`, `News_flag`

Nota: `Date` es opcional para el pipeline (se usa para alinear series y fechas en gráficos/salidas). `Adj Close` debe estar en las features porque es el objetivo de predicción.


## Modelo y pipeline (PyTorch)

Archivo: `clean_new_model/lstm_pipeline.py`

- Arquitectura: `LSTMPredictor(input_dim, hidden_dim=64, num_layers=1, dropout=0.0)` + `Linear(hidden_dim → 1)`.
- Sliding window: para una ventana de longitud `seq`, la entrada es `seq-1` timesteps y la etiqueta es `Adj Close` en el paso `seq`.
- Escalado por ticker con `MinMaxScaler` SOLO para las columnas indicadas en `features_to_scale`. 
- Split temporal por archivo: `split` fracción para train y el resto para test. Opción de validación temporal a cola de train con `val_frac` para early stopping.
- Métricas por ticker y globales: MAE, MSE, NRMSE (normalizada por std de y_true), R2, MAPE, WMAPE, y HIT_RATE (dirección del movimiento).


## Dependencias

Core (modelo/experimentación):

```bash
pip install torch numpy pandas scikit-learn
```

Noticias/opcional (depende de tu entorno y GPU/CPU):

- Hugging Face datasets/transformers para FinBERT (`ProsusAI/finbert`).
- Ollama para la recomendación local (ver documentación de Ollama e instalar el modelo usado en `recomendacion_ollama.ipynb`).


## Ejemplo de uso 

Desde un notebook (p.ej. `experiments_clean.ipynb`) o un script Python:

```python
from clean_new_model.lstm_pipeline import run_experiment

data_dir = "clean_new_model/dataset_ceci_sol"
tickers = ["AAPL", "MSFT", "NVDA"]           # tickers disponibles como CSV dentro de data_dir
test_tickers = ["AAPL"]                      # opcional: subset de evaluación

# Tiene que incluir 'Adj Close' como target
features = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume", "VIX",
    "puntaje_sent_fin_pos", "puntaje_sent_fin_neu", "puntaje_sent_fin_neg",
    "recom_inv_llm", "rows_per_day", "has_news", "Scaled_sentiment", "News_flag",
]

# Escalar solo las columnas que tienen sentido; puedes dejar vacío si no deseas escalar nada
features_to_scale = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume", "VIX",
    "puntaje_sent_fin_pos", "puntaje_sent_fin_neu", "puntaje_sent_fin_neg",
    "recom_inv_llm", "rows_per_day", "Scaled_sentiment",
]

result = run_experiment(
    data_dir=data_dir,
    features=features,
    features_to_scale=features_to_scale,
    tickers=tickers,
    test_tickers=test_tickers,   # o None para evaluar en todos los tickers de train
    seq=50,                      # longitud de ventana
    split=0.85,                  # fracción de train
    batch_size=32,
    epochs=20,
    hidden_dim=64,
    num_layers=1,
    dropout=0.0,
    lr=1e-3,
    device="cuda",               # "cuda" si hay GPU disponible, sino "cpu"
    seed=42,
    val_frac=0.1,                # para early stopping
    early_stopping=True,
    patience=5,
    min_delta=0.0,
)

# Diccionario con:
# - result["global_metrics"], result["weighted_metrics"]
# - result["per_ticker"] (métricas por ticker)
# - result["per_ticker_series"] (series originales para graficar: fechas, y_true, y_pred)
```


## Cómo reproducir los experimentos del informe

1) Asegúrate de tener los CSVs por ticker en `clean_new_model/dataset_ceci_sol/`.
2) Abrir `clean_new_model/experiments_clean.ipynb` y configura:
   - Lista de `tickers` para entrenar/evaluar.
   - `features` y `features_to_scale` (incluyendo siempre `Adj Close`).
   - Hiperparámetros (`seq`, `epochs`, `hidden_dim`, etc.).
3) Ejecuta las celdas para entrenar y evaluar. Los resultados por ticker pueden exportarse a `clean_new_model/dfs_por_ticker_csv/`.


## Notas 

- Evitar fuga de datos: el escalado se ajusta solo en train por ticker y se aplica después a test.
- Reproducibilidad: se fijan semillas en Python, NumPy, y PyTorch; CUDNN en modo determinista.
- HIT_RATE: evalúa dirección del cambio con respecto al valor previo; útil como métrica de señal.
- Ventaneo: si una serie/ticker es muy corta, el split de validación puede desactivarse (ajustar `val_frac`).


## Referencias

- Dataset FNSPID: `https://huggingface.co/datasets/Zihan1004/FNSPID`
- FinBERT (ProsusAI): `https://huggingface.co/ProsusAI/finbert`
- Ollama: `https://ollama.com/`


