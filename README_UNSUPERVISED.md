# Análisis de Clustering con K-Means sobre el dataset de Scoring Crediticio

1. Introducción

Como parte del desarrollo de un sistema de scoring crediticio, se incorpora una técnica de aprendizaje no supervisado basada en K-Means.
El propósito es complementar el modelo supervisado principal mediante la detección de grupos de clientes con características similares, tanto socioeconómicas como de comportamiento financiero.

Con este análisis se busca:

Detectar patrones naturales en la población.

Comparar diferencias entre segmentos encontrados.

Identificar grupos que presenten comportamientos relevantes para la gestión y evaluación del riesgo.

Todo el trabajo se realiza exclusivamente sobre el dataset de entrenamiento, garantizando la ausencia de data leakage.

2. Técnica Utilizada y Motivo de la Selección

2.1. Algoritmo empleado: K-Means

K-Means es un método de clustering que:

Agrupa los registros en K clusters.

Minimiza la distancia de cada observación respecto a su centroide.

Produce grupos compactos formados por clientes con características similares.

2.2. Razones de su elección

Se eligió K-Means debido a que:

Es un algoritmo simple, rápido y fácil de interpretar.

Permite segmentar perfiles de clientes de forma objetiva.

Funciona especialmente bien con variables numéricas escaladas.

Facilita la búsqueda del número óptimo de clusters mediante:

El método del codo (inercia)

El índice de Silhouette

3. Datos Utilizados

3.1. Archivo procesado

El análisis se realizó sobre el archivo:

data/application_.parquet


El dataset contiene:

307.511 registros

122 columnas

3.2. Variables consideradas para el clustering

Se utilizaron las siguientes columnas numéricas presentes en el archivo:
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
EXT_SOURCE_1
EXT_SOURCE_2
EXT_SOURCE_3
DAYS_BIRTH
DAYS_EMPLOYED

La columna TARGET se usó únicamente para depuración (eliminación de registros incompletos), no para entrenar el modelo no supervisado.

3.3. Preparación del dataset

Las principales acciones fueron:

Selección de columnas numéricas relevantes.
Eliminación de filas con valores faltantes.
Normalización mediante StandardScaler.

4. Metodología (CRISP-DM)

4.1. Business Understanding

El objetivo es identificar segmentos reales de clientes en función de sus atributos.
Estos clusters pueden relacionarse con diferentes perfiles de riesgo o comportamiento financiero.
4.2. Data Understanding

Se inspeccionó estructura, tamaños y tipos de datos.
Se detectaron y trataron valores faltantes.
Se validó la coherencia e integridad de las variables seleccionadas.

4.3. Data Preparation

Selección final de variables numéricas.
Limpieza de registros incompletos.
Escalamiento previo al uso de K-Means.

4.4. Modeling

Se evaluaron valores de K entre 2 y 7, obteniendo:

K	Inercia	Silhouette
2	783743	0.2342
3	646544	0.2757
4	571386	0.2067
5	553215	0.2011
6	520018	0.1833
7	486410	0.1787
4.5. Selección del K óptimo

El gráfico del codo muestra un punto de quiebre entre K=3 y K=4.

El índice de Silhouette más alto corresponde a K=3.
K óptimo seleccionado: 3 clusters.

4.6. Evaluation

Se validó cohesión y separación de los clusters.
Los resultados son consistentemente razonables para un dataset financiero.
Los clusters formados representan patrones relevantes.
4.7. Deployment

El clustering puede aprovecharse para:
Crear un nuevo atributo: cluster_id.
Realizar análisis y segmentación de riesgo.
Generar grupos para estrategias comerciales o de control.

5. Instrucciones de Ejecución
5.1. Instalar dependencias
pip install pandas numpy scikit-learn matplotlib pyarrow


o, como en este proyecto:

py -3.13 -m pip install pandas numpy scikit-learn matplotlib pyarrow

5.2. Ejecutar el script

Con Python 3.13:

py -3.13 src/ea3_unsupervised_kmeans.py


O en general:

python src/ea3_unsupervised_kmeans.py


El script mostrará:

Información del dataset

Valores de inercia y silhouette

Gráfico del codo

Gráfico de silhouette

Resultado con los clusters finales

6. Estructura del Proyecto
EA3_Scoring/
│
├── data/
│   └── application_.parquet
│
├── src/
│   └── ea3_unsupervised_kmeans.py
│
└── README_UNSUPERVISED.md

7. Conclusión

El dataset presenta una estructura óptimamente representada por 3 clusters.

El índice de silhouette (~0.27) evidencia una segmentación adecuada.

El clustering agrega valor para análisis posteriores y estrategias de riesgo.