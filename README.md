# Modelo 1 — Predicción del estado de una carga

Árbol de decisión para predecir `Status`, con una interfaz en Streamlit basada en el `app.py` de referencia.

## Archivos

| Archivo | Uso |
| --- | --- |
| `modelo_1.py` | Entrena el modelo original y genera `modelo_1.pkl` en tu equipo. |
| `app.py` | Carga el modelo y permite ingresar los datos para predecir. |
| `requirements.txt` | Dependencias para entrenamiento y despliegue. |
| `README.md` | Instrucciones del proyecto. |

El CSV y el `.pkl` no están incluidos: el modelo se genera con tu propio `cleaned_data_shipment.csv`.

## 1. Instalar

Utiliza **Python 3.12**. Abre una terminal en esta carpeta; preferiblemente trabaja en un entorno virtual.

```bash
python -m pip install -r requirements.txt
```

## 2. Generar el modelo en local

Coloca `cleaned_data_shipment.csv` junto a `modelo_1.py`, con las mismas columnas y datos preparados que utilizabas en el código original. Ejecuta:

```bash
python modelo_1.py --solo-exportar
```

También puedes indicar otra ruta:

```bash
python modelo_1.py "C:/ruta/cleaned_data_shipment.csv" --solo-exportar
```

Al terminar se crea **`modelo_1.pkl` junto a `modelo_1.py`**. Como se ejecuta en tu equipo, el archivo ya queda guardado localmente. Contiene el árbol entrenado, las columnas de entrada, las opciones del formulario, los nombres de los estados y el error sobre el conjunto de prueba.

`--solo-exportar` termina después de guardar el modelo y evita ejecutar los gráficos y evaluaciones posteriores. Para ejecutar también esos bloques originales, utiliza `python modelo_1.py`. Cierra la ventana del árbol para que el script continúe. La matriz de confusión y la curva ROC originales asumen dos estados; si tu CSV tiene más, utiliza `--solo-exportar`.

## 3. Probar Streamlit

```bash
python -m streamlit run app.py
```

El formulario toma los nombres, el orden y las categorías reales guardadas durante el entrenamiento. Las variables numéricas se ingresan con `st.number_input` y las categóricas con `st.selectbox`. La predicción se actualiza al cambiar los valores.

## 4. Desplegar desde GitHub

Sube estos cuatro archivos a la misma carpeta del repositorio, como en la estructura de referencia:

- `README.md`
- `app.py`
- `modelo_1.pkl` generado en el paso 2
- `requirements.txt`

`modelo_1.py` es opcional en el repositorio; el CSV no se necesita para ejecutar la app.

En [Streamlit Community Cloud](https://share.streamlit.io/), crea una app, selecciona el repositorio y su rama, y elige **`app.py`** como archivo principal. En la configuración avanzada selecciona **Python 3.12**, igual que en local. Usa este mismo `requirements.txt` al entrenar y desplegar para mantener las versiones compatibles con el `.pkl`.

## Cambios realizados

- Se conservan la preparación de datos, las variables predictoras, las dummies del entrenamiento, la división estratificada 70/30 y el árbol con `criterion='gini'`, `min_samples_leaf=50` y `max_depth=10`.
- Se adapta la ruta de Colab a un archivo local y se sustituye la descarga exclusiva de Colab por el guardado local.
- Se agrega la exportación justo después de `fit`, antes de que la validación cruzada vuelva a crear `modelTree`.
- `app.py` sigue el orden de la referencia: cargar modelo, capturar datos, crear DataFrame, generar dummies, alinear columnas y predecir. En despliegue usa `drop_first=False` y `reindex`, como el ejemplo.
- El resultado muestra el nombre de `Status`. El aviso utiliza el error de clasificación del modelo entrenado sobre el 30% de prueba.

Se mantiene la división aleatoria del original, sin añadir `random_state`; los resultados pueden variar al volver a entrenar.

Documentación: [despliegue en Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) y [persistencia de modelos de scikit-learn](https://scikit-learn.org/stable/model_persistence.html).
