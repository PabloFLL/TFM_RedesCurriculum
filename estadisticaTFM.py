import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chi2_contingency

# Rutas de los archivos CSV
archivo_csv1 = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\programa_crear_redes\red_fisica_matematicas\matriz_adjacente_pablo_conjunta_actualizado.csv"
archivo_csv2 = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\programa_crear_redes\red_fisica_matematicas\matriz_adjacente_pablo_conjunta_actualizado.csv"

# Leer los CSVs usando pandas
datos_csv1 = pd.read_csv(archivo_csv1)
datos_csv2 = pd.read_csv(archivo_csv2)

# Filtrar solo las columnas numéricas para cada DataFrame
datos_numericos_csv1 = datos_csv1.select_dtypes(include=['number'])
datos_numericos_csv2 = datos_csv2.select_dtypes(include=['number'])

# Mostrar las primeras filas de los datos numéricos
#print("Primeros datos numéricos del archivo 1:")
#print(datos_numericos_csv1.head(20))

#print("\nPrimeros datos numéricos del archivo 2:")
#print(datos_numericos_csv2.head(20))

# Unir los datos numéricos si es necesario
datos_numericos_combinados = pd.concat([datos_numericos_csv1, datos_numericos_csv2])

# Mostrar los datos combinados
#print("\nDatos numéricos combinados:")
#print(datos_numericos_combinados.head(20))

# Exportar los datos combinados a un archivo CSV para ver si todo va bien
#archivo_csv_export = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\programa_crear_redes\red_fisica\datos_numericos_combinados.csv"  # Ruta donde se guardará el archivo CSV
#datos_numericos_combinados.to_csv(archivo_csv_export, index=False)  # Guardar como archivo CSV sin índices
#print(f"Datos combinados exportados a CSV: {archivo_csv_export}")

# Exportar los datos combinados a un archivo Excel
#archivo_excel_export = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\programa_crear_redes\red_fisica\datos_numericos_combinados.xlsx"  # Ruta donde se guardará el archivo Excel
#datos_numericos_combinados.to_excel(archivo_excel_export, index=False)  # Guardar como archivo Excel sin índices
#print(f"Datos combinados exportados a Excel: {archivo_excel_export}")

# Mostrar las dimensiones de los DataFrames
print(f"Dimensiones del archivo 1 (filas, columnas): {datos_numericos_csv1.shape}")
print(f"Dimensiones del archivo 2 (filas, columnas): {datos_numericos_csv2.shape}")
print(f"Dimensiones del archivo combinado (filas, columnas): {datos_numericos_combinados.shape}")



# APLICAMOS KAPPA DE COHEN

# Convertir todas las columnas numéricas en listas y unirlas en una lista única
lista_Pablo = []
lista_Paula = []

# Para el primer archivo CSV
for col in datos_numericos_csv1.columns:
    lista_Pablo.extend(datos_numericos_csv1[col].dropna().tolist())  # Añadir cada columna como lista, ignorando NaN

# Para el segundo archivo CSV
for col in datos_numericos_csv2.columns:
    lista_Paula.extend(datos_numericos_csv2[col].dropna().tolist())  # Añadir cada columna como lista, ignorando NaN

# Mostrar la lista completa
print(lista_Pablo)
print(len(lista_Pablo))
print(lista_Paula)
print(len(lista_Paula))
# Aplicar el cálculo de Kappa de Cohen
kappa = cohen_kappa_score(lista_Pablo, lista_Paula)

# Mostrar el resultado de Kappa de Cohen
print(f"Kappa de Cohen: {kappa}")

# Contar cuántos elementos son iguales en la misma posición
iguales = sum(1 for x, y in zip(lista_Pablo, lista_Paula) if x == y)
diferentes = sum(1 for x, y in zip(lista_Pablo, lista_Paula) if x != y)
print(f"Los elementos iguales son: {iguales}")
print(f"Los elementos diferentes son: {diferentes}")



# APLICAMOS CHI-CUADRADO

# Crear un DataFrame con ambas listas
df = pd.DataFrame({'Lista1': lista_Pablo, 'Lista2': lista_Paula})

# Crear la tabla de contingencia
tabla_contingencia = pd.crosstab(df['Lista1'], df['Lista2'])

# Mostrar la tabla de contingencia
print(tabla_contingencia)
chi2, p_valor, dof, esperado = chi2_contingency(tabla_contingencia)

# Mostrar los resultados
print(f"Valor Chi-cuadrado: {chi2}")
print(f"Valor p: {p_valor}")
print(f"Grados de libertad: {dof}")
print(f"Frecuencias esperadas:\n{esperado}")