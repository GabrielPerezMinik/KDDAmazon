import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carga de datos
df = pd.read_csv('amazon.csv')

print(df.isnull().sum())
print(df.select_dtypes(include="object").apply(lambda x: x.str.strip().eq("").sum()))

# Seleccion de los datos relevantes  a Minar
df_selected = df[['product_id', 'rating', 'discounted_price', 'actual_price', 'review_content']].copy()

## Limpieza de los datos
# Eliminar filas duplicadas
df_cleaned = df_selected.drop_duplicates().copy()

print("Filas antes de eliminar NaNs:", df_cleaned.shape[0])

print("Filas después de eliminar NaNs:", df_cleaned.shape[0])

print(df_selected['actual_price'].unique()[:10])  # Muestra algunos valores

df_cleaned.loc[:, 'actual_price'] = df_cleaned['actual_price'].str.replace(r'[^\d.]', '', regex=True)
df_cleaned.loc[:, 'discounted_price'] = df_cleaned['discounted_price'].str.replace(r'[^\d.]', '', regex=True)

# Conversión a tipo numérico
df_cleaned.loc[:, 'actual_price'] = pd.to_numeric(df_cleaned['actual_price'], errors='coerce')
df_cleaned.loc[:, 'discounted_price'] = pd.to_numeric(df_cleaned['discounted_price'], errors='coerce')

print(df_cleaned[['actual_price', 'discounted_price']].isnull().sum())

# Eliminar filas con precios nulos
df_cleaned = df_cleaned.dropna(subset=['discounted_price', 'actual_price'])

# Crear una nueva columna con el margen de descuento
df_cleaned['discount_margin'] = (df_cleaned['actual_price'] - df_cleaned['discounted_price']) / df_cleaned['actual_price']

# Crear una columna de rango de precio
df_cleaned['price_range'] = pd.cut(df_cleaned['actual_price'], bins=[0, 50, 100, 200, 500, 1000], labels=['Bajo', 'Medio', 'Alto', 'Muy Alto', 'Lujo'])

# Seleccionar características relevantes para el clustering
df_clustering = df_cleaned[['actual_price', 'rating']]

print(df_clustering.dtypes)

print(df_clustering['rating'].unique())

df_cleaned['rating'] = pd.to_numeric(df_cleaned['rating'], errors='coerce')
df_clustering = df_cleaned[['actual_price', 'rating']].dropna()
print(df_clustering['rating'].unique())

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
## Transformación del conjunto de datos de entrada
# Normalizar los datos

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

df_cleaned = df_cleaned.loc[df_clustering.index].copy()
df_cleaned['cluster'] = clusters

# Ver los resultados
print(df_cleaned.groupby('cluster').agg({'actual_price': 'mean', 'rating': 'mean'}))
# Evaluar el número de productos por cluster
print(df_cleaned['cluster'].value_counts())

upper_bound = df_cleaned['actual_price'].quantile(0.99)
df_cleaned_no_outliers = df_cleaned[df_cleaned['actual_price'] <= upper_bound]

# Agrupar sin outliers
print(df_cleaned_no_outliers.groupby('cluster').agg({'actual_price': 'mean', 'rating': 'mean'}))

# Analizar las características de cada cluster
cluster_summary_no_outliers = df_cleaned_no_outliers.groupby('cluster').agg({
    'actual_price': ['mean', 'std'],
    'rating': ['mean', 'std']
})
print(cluster_summary_no_outliers)

## Interpretación y evaluación de datos
# Visualización del precio medio y rating por cluster
df_cleaned.groupby('cluster').agg({'actual_price': 'mean', 'rating': 'mean'}).plot(kind='bar')
plt.title('Precio medio y Rating por Cluster')
plt.ylabel('Valor')
plt.show()

profile = ProfileReport(df_cleaned_no_outliers, title='Reporte Exploratorio - Amazon Reviews', explorative=True)
profile.to_file('reporte_amazon.html')