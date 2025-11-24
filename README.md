# Text2Market
Señales de noticias financieras para forecasting diario de volatilidad y retornos

## Archivos incluidos en el repo

### read_data_news.ipynb
- Lectura del dataset proveniente del paper de FINSPID (descargado de https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/Stock_news/nasdaq_exteral_data.csv) y análisis y filtrado inicial de la información.

### eda_y_seleccion_sub_dataset.ipynb
- Lectura del subdataset obtenido en el notebbok anterior y análisis de cantidad de noticias por ticker y en el tiempo.
- Selección del subset de tickers y período de tiempo para trabajar en los siguientes steps del proyecto.

### deteccion_tickers.ipynb
- Pipeline para la detección de tickers (de un listado preseleccionado) presentes de forma explícita en las noticias a partir de un patrón Regex.

### deteccion_sent_financiero_con_finbert.ipynb
- Clasificación de noticias según la cantidad de tokens
- Pipeline para la extracción de los puntajes de sentimiento financiero (positivo, negativo y neutral) desde **ProsusAI/finbert** para las noticias con 512 tokens o menos (incluyendo caracteres especiales)

### recomendacion_ollama.ipynb
- Pipeline para obtener recomendación de inversión con Ollama (modelo: martain7r/finance-llama-8b:fp16) mediante un prompt que devuelve un valor de 1 a 5 para cada ticker/notica donde:
```
1 = Strongly Negative (Do not invest)
3 = Neutral
5 = Strongly Positive (Strong Buy)
```
