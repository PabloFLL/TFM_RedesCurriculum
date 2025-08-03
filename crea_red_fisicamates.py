import numpy as np
import random
import os
import json

# Función para cargar palabras desde un archivo
def cargar_palabras(archivo):
    with open(archivo, 'r', encoding='utf-8') as f:
        palabras = f.read().strip().split('\n')
    return palabras

# Función para cargar el progreso de un archivo
def cargar_progreso(usuario):
    # Intentamos cargar el archivo de progreso, si existe
    progreso_path = f'progreso_{usuario}.json'
    if os.path.exists(progreso_path):
        with open(progreso_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return []

# Función para guardar el progreso en un archivo
def guardar_progreso(usuario, progreso):
    with open(f'progreso_{usuario}.json', 'w', encoding='utf-8') as f:
        json.dump(progreso, f)

# Función para cargar el nombre de usuario desde un archivo
def cargar_usuario():
    usuario_path = 'usuario.txt'
    if os.path.exists(usuario_path):
        with open(usuario_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        return None

# Función para guardar el nombre de usuario en un archivo
def guardar_usuario(usuario):
    with open('usuario.txt', 'w', encoding='utf-8') as f:
        f.write(usuario)

# Función para construir la matriz de adyacencia
def construir_matriz(palabras, progreso):
    n = len(palabras)
    matriz = np.full((n, n), 'NA')  # Inicializar matriz con 'NA'

    # Recuperar las posiciones y valores del progreso guardado
    for i, j, valor in progreso:
        matriz[i][j] = valor

    # Generar todas las posiciones válidas que no han sido clasificadas
    posiciones_validas = [(i, j) for i in range(n) for j in range(i + 1, n)]
    posiciones_clasificadas = {(i, j) for i, j, _ in progreso}
    posiciones_por_clasificar = [pos for pos in posiciones_validas if pos not in posiciones_clasificadas]
    
    if not posiciones_por_clasificar:
        print("¡Ya has clasificado todas las combinaciones!")
        return matriz

    # Barajar las posiciones restantes para aleatoriedad
    random.shuffle(posiciones_por_clasificar)

    contador = len(posiciones_clasificadas)
    for i, j in posiciones_por_clasificar:
        while True:
            try:
                print(f"Palabras clasificadas: {contador}")
                valor = input(f"¿Cómo se relacionan los siguientes conceptos? \n 0: no relación, 1: primera engloba segunda, 2: segunda engloba primera, 3: relacionadas pero sin englobarse, q: salir \n \n CONCEPTO 1: \n '{palabras[i]}' \n \n CONCEPTO 2: \n '{palabras[j]}' \n \n CLASIFICA: ")
                
                if valor.lower() == 'q':
                    print("Ejecutando la salida y guardando la matriz.")
                    return matriz  # Regresar la matriz guardada

                if valor in ['0', '1', '2', '3']:
                    matriz[i][j] = int(valor)
                    progreso.append((i, j, int(valor)))  # Guardamos el par y su valor
                    contador += 1
                    print()
                    break
                else:
                    print("Por favor, introduce un valor válido: 0, 1, 2 o 3.")
            except ValueError:
                print("Entrada no válida. Por favor, introduce un número o 'q' para salir.")

    return matriz

# Función principal
def main():
    # Intentamos cargar el nombre de usuario
    usuario = cargar_usuario()

    if not usuario:
        # Si no se ha encontrado el nombre de usuario, pedimos uno nuevo
        usuario = input("Introduce tu nombre de usuario: ")
        guardar_usuario(usuario)  # Guardamos el nombre para futuras ejecuciones
    
    print(f"Biensvenido de nuevo, {usuario}!")

    # Cargar palabras desde el archivo
    nombre_archivo = 'fisica_matematicas.txt'  # Cambia esto por el nombre de tu archivo
    palabras = cargar_palabras(nombre_archivo)

    # Cargar el progreso de clasificación desde un archivo (si existe)

    progreso = cargar_progreso(usuario)


    # Construir la matriz de adyacencia, pasando el progreso
    matriz_adjacente = construir_matriz(palabras, progreso)

    # Guardar el progreso actualizado en JSON
    guardar_progreso(usuario, progreso)
    
    # Reconstruir la matriz completa desde el progreso guardado en JSON
    n = len(palabras)
    matriz_completa = np.full((n, n), 'NA', dtype=object)

    for i, j, valor in progreso:
        matriz_completa[i][j] = valor

    # Guardar los resultados en un archivo CSV usando pandas
    import pandas as pd
    df = pd.DataFrame(matriz_adjacente, index=palabras, columns=palabras)
    df.to_csv(f'matriz_adjacente_{usuario}.csv')

    print(f"Matriz de adyacencia guardada en 'matriz_adjacente_{usuario}.csv'.")

# Ejecutar el programa
if __name__ == "__main__":
    main()

