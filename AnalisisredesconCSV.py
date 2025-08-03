import csv
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import community as community_louvain
from collections import defaultdict
import matplotlib.cm as cm
from infomap import Infomap
import scipy as sp
import random
from collections import defaultdict, Counter
from networkx.drawing import nx_agraph
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, average
import igraph as ig
import math
from itertools import combinations
from networkx.drawing.nx_agraph import graphviz_layout

# Solicitar la ruta del archivo CSV antes de ejecutar
archivo_csv = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\programa_crear_redes\red_matematicas\matriz_adjacente_pablo_mod.csv"

def cargar_grafo_desde_csv(ruta_csv):
    """Carga un grafo mixto desde un CSV con nodos dirigidos y no dirigidos."""
    if not os.path.exists(ruta_csv):
        print(f" Error: El archivo {ruta_csv} no existe.")
        return None

    with open(ruta_csv, 'r', newline='', encoding='utf-8') as file:
        reader = list(csv.reader(file))

    reader = [fila for fila in reader if any(fila)]  # 🔹 Eliminar filas vacías
    nombres_nodos = reader[0]  # 🔹 Extraer nombres de nodos

    G = nx.DiGraph()  # 🔹 Usamos un DiGraph pero añadimos aristas no dirigidas manualmente
    Gno = nx.Graph()  # 🔹 Usamos un DiGraph pero añadimos aristas no dirigidas manualmente
    Gtotal = nx.DiGraph()
    Gtotalsimple =  nx.Graph()


    for fila in reader[1:]:
        if not fila:
            continue

        nodo_origen = fila[0]
        relaciones = fila[1:]

        relaciones = [int(valor) if valor.strip().isdigit() else 0 for valor in relaciones]

        for i, valor in enumerate(relaciones):
            nodo_destino = nombres_nodos[i + 1]

            # ⚠️ Evitar autoenlaces y valores 0
            if nodo_origen == nodo_destino or valor == 0:
                continue

            # 🔹 Agregar enlaces según las reglas
            if valor == 1:
                G.add_edge(nodo_origen, nodo_destino, tipo="dirigido")  # Dirigido de A → B
                Gtotal.add_edge(nodo_origen, nodo_destino, tipo="dirigido")  # Dirigido de A → B
                Gtotalsimple.add_edge(nodo_origen, nodo_destino, tipo="simple")
            elif valor == 2:
                G.add_edge(nodo_destino, nodo_origen, tipo="dirigido")  # Dirigido de B → A
                Gtotal.add_edge(nodo_destino, nodo_origen, tipo="dirigido")  # Dirigido de A → B
                Gtotalsimple.add_edge(nodo_destino, nodo_origen, tipo="simple")
            elif valor == 3:
                    Gno.add_edge(nodo_origen, nodo_destino, tipo="no dirigido")  # Agregar solo una vez
                    Gtotal.add_edge(nodo_origen, nodo_destino, tipo="dirigido")
                    Gtotal.add_edge(nodo_destino, nodo_origen, tipo="dirigido")# Dirigido de A → B
                    Gtotalsimple.add_edge(nodo_origen, nodo_destino, tipo="simple")

    return G,Gno,Gtotal, Gtotalsimple

def clean_name(name):
    return name.split()[0]

def dibujar_grafo(G, Gsi, Gno, color, titulo):
    """Dibuja y guarda el grafo mixto con colores diferenciando aristas dirigidas y no dirigidas."""
    plt.figure(figsize=(19.2, 10.8))
    #color = color.reindex(sorted(G.nodes()))
    # 🔹 Generar UNA única disposición de nodos para TODOS los grafos
    pos = nx.kamada_kawai_layout(G)

    # 🔹 Dibujar nodos con la misma disposición
    sc = nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=color, edgecolors="black", cmap=plt.cm.coolwarm)

    # 🔹 Dibujar aristas dirigidas con flechas
    edges_dirigidos = [(u, v) for u, v, d in Gsi.edges(data=True) if d.get("tipo") == "dirigido"]
    nx.draw_networkx_edges(Gsi, pos, edgelist=edges_dirigidos, edge_color="green", style="solid", arrows=True,
                           arrowsize=15, width=2)

    # 🔹 Dibujar aristas no dirigidas sin flechas
    # edges_no_dirigidos = [(u, v) for u, v, d in Gno.edges(data=True) if d.get("tipo") == "no dirigido"]
    # nx.draw_networkx_edges(G, pos, edgelist=edges_no_dirigidos, edge_color="red", style="dashed", arrows=False, width=2)

    # 🔹 Dibujar etiquetas
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="black",font_weight="light", verticalalignment='center',horizontalalignment='center')

    # 🔹 Agregar barra de color
    cbar = plt.colorbar(sc, shrink=0.8)
    cbar.set_label(titulo, fontsize=12)


    # 🔹 Guardar con el nombre basado en título
    nombre_archivo = f"{titulo.replace(' ', '_').lower()}.png"
    plt.title(titulo)
    plt.savefig(nombre_archivo, format="png", dpi=300)

    plt.show()

    print("\n✅ La imagen del grafo ha sido guardada como 'grafo_mixto.png'.")





def main():
    grafodirigido,grafonodirigido,grafototal,grafototalsimple = cargar_grafo_desde_csv(archivo_csv)

    if grafodirigido is None:
        return




# #Estudio del camino medio, clustering coefficient y diametro
#     longitud_media_camino = nx.average_shortest_path_length(grafototal)
#     print(f"La longitud del camino medio es: {longitud_media_camino:.2f}")
#     diametro = nx.diameter(grafototal)
#     print(f"El diametro del camino medio es: {diametro}")
#     clustering = nx.average_clustering(grafototal)
#     print(f"El clustering del camino medio es: {clustering:.2f}")
#     numero_enlaces = grafototalsimple.number_of_edges()
#     print(f"Hay {numero_enlaces} enlaces")
#
#
# # #Estudio del grado
#     print("Estudiamos el grado del grafo:\n")
#     grado_total = pd.Series(dict(grafototal.degree())) #Cargo el grado en una variable y lo ordeno
#
#     pd.set_option('display.max_rows', None)  # Mostrar todas las filas
#     print("Estudiamos la centralidad del grafo:")
#    # print("\n Ordenados todos los elementos de la media:\n", grado_total.sort_values(ascending=False))
#     print("\n-----------------------------------")
#
#     #dibujar_grafo(grafototal, grafototal, grafototal, grado_total, titulo="Grado")

#-----------------------------------------------------------------------
#     #Estudio de la centralidad

    # centralidad_grado_dirigido = pd.Series(nx.degree_centrality(grafodirigido)) #Centralidad del dirigido
    # centralidad_grado_nodirigido = pd.Series(nx.degree_centrality(grafonodirigido)) #Centralidad del no dirigido
    # centralidad_grado_nodirigido, centralidad_grado_dirigido = centralidad_grado_nodirigido.align(centralidad_grado_dirigido, join="outer") #Los alineo para que todo sea correcto
    #
    # media_centralidad=(centralidad_grado_dirigido+centralidad_grado_nodirigido)/2 #Media de los valores
    # media_centralidad_normalizada = media_centralidad / media_centralidad.max()
    # media_centralidad_normalizada = media_centralidad_normalizada.reindex(grafototal.nodes(), fill_value=0) #SIRVE PARA REORDENAR Y QUE PINTE BIEN

#Caida de centralidad
    # 1. Calcular la centralidad
    centralidad_buena = pd.Series(nx.degree_centrality(grafototal))

    # 2. Normalizar y ordenar
    media_centralidad_sorted = centralidad_buena.sort_values(ascending=False) / centralidad_buena.max()

    # 3. Leer el Excel
    ruta_excel = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\matesidentificador.xlsx"  # ← reemplaza con la ruta correcta
    df_ids = pd.read_excel(ruta_excel)

    # Asegurar formato limpio para evitar errores de coincidencia
    df_ids['Saber'] = df_ids['Saber'].str.strip()
    df_ids['Identificador'] = df_ids['Identificador'].astype(str).str.strip()

    # 4. Crear diccionario {nombre_largo: id_corto}
    mapa_ids = dict(zip(df_ids['Saber'], df_ids['Identificador']))

    # 5. Reemplazar nombres largos por los identificadores en el eje X
    x_labels = media_centralidad_sorted.index.map(lambda nombre: mapa_ids.get(nombre.strip(), nombre))

    # 6. Graficar
    plt.figure(figsize=(19.2, 10.8))
    plt.scatter(x_labels, media_centralidad_sorted.values, color='blue', alpha=0.7)

    plt.xlabel("Identificador de los saberes básicos", fontsize=16)
    plt.ylabel("Grado normalizado", fontsize=16)
    plt.title("Caída del grado en la red de Matemáticas", fontsize=16)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 7. Guardar imagen
    ruta_guardado = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\imagenes\caidacentralidad_mates"
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")

    plt.show()



   #----------------------------------------------------------------
    # # Estudio de la betweenness

#
#     betweenness_grado_dirigido = pd.Series(nx.betweenness_centrality(grafodirigido))  # Centralidad del dirigido
#     betweenness_grado_nodirigido = pd.Series(nx.betweenness_centrality(grafonodirigido))  # Centralidad del no dirigido
#     betweenness_grado_nodirigido, betweenness_grado_dirigido = betweenness_grado_nodirigido.align(betweenness_grado_dirigido, join="outer")  # Los alineo
#
#     media_betweenness = (betweenness_grado_dirigido+betweenness_grado_nodirigido)/2  # Media de los valores
#     media_betweenness = media_betweenness/media_betweenness.max()
#     media_betweenness = media_betweenness.reindex(grafototal.nodes(), fill_value=0)
#
#     bet_buena = pd.Series(nx.betweenness_centrality(grafototal))
#
#     pd.set_option('display.max_rows', None)  # Mostrar todas las filas
#     print("Estudiamos la betweenness del grafo:")
#     #print("\n Ordenados todos los elementos de la media:\n", bet_buena.sort_values(ascending=False))
#     print("\n-----------------------------------")
#
#     #dibujar_grafo(grafototal, grafototal, grafototal, bet_buena, titulo="Betweenness")
#
#
#
#     #dibujar_grafo(grafototal, grafo, grafonodirigido, media_betweenness, titulo="Betweenness")
#
#
#    #  # ----------------------------------------------------------------
#    #  # Estudio de la closeness
#     closeness_grado_dirigido = pd.Series(nx.closeness_centrality(grafodirigido))  # Centralidad del dirigido
#     closeness_grado_nodirigido = pd.Series(nx.closeness_centrality(grafonodirigido))  # Centralidad del no dirigido
#     closeness_grado_nodirigido, closeness_grado_dirigido = closeness_grado_nodirigido.align(closeness_grado_dirigido, join="outer")  # Los alineo
#
#     media_closeness = (closeness_grado_dirigido+closeness_grado_nodirigido)/2  # Media de los valores
#     media_closeness = media_closeness/media_closeness.max()
#     media_closeness = media_closeness.reindex(grafototal.nodes(), fill_value=0)
#
#     close_buena = pd.Series(nx.closeness_centrality(grafototal))
#
#     pd.set_option('display.max_rows', None)  # Mostrar todas las filas
#     print("Estudiamos la closeness del grafo:")
#    # print("\n Ordenados todos los elementos de la media:\n", close_buena.sort_values(ascending=False))
#     print("\n-----------------------------------")
#
#     #dibujar_grafo(grafototal, grafototal, grafototal, close_buena, titulo="Closeness")
#
# # ----------------------------------------------------------------
# # ----------------------------------------------------------------
#
#     #METACENTRALIDAD
#
#     # Ordenar los nodos según cada centralidad
#
#     centralidad_normalizada=centralidad_buena/centralidad_buena.max()
#     #print(centralidad_normalizada.sort_values(ascending=False))
#     print("\n-----------------------------------")
#
#     bet_normalizada=bet_buena/bet_buena.max()
#     #print(bet_normalizada.sort_values(ascending=False))
#     print("\n-----------------------------------")
#
#     close_normalizada=close_buena/close_buena.max()
#     #print(close_normalizada.sort_values(ascending=False))
#     print("\n-----------------------------------")
#
#     metacentralidad = centralidad_normalizada + bet_normalizada + close_normalizada
#     print("La metacentralidad es:")
#     #print(metacentralidad.sort_values(ascending=False))
#
# # PROBAMOS A ELIMINAR ALGUNOS NODOSSSS
#
#     # Ordenar los nodos por grado (de mayor a menor)
#     sorted_nodes = sorted(metacentralidad.items(), key=lambda x: x[1], reverse=True)
#
#     # Seleccionar los nodos más conectados (por ejemplo, eliminar los primeros 10 nodos más conectados) 11 FISMAT, 6 FIS, 4 MAT
#     num_nodes_to_remove = 11
#     nodes_to_remove = [node for node, degree in sorted_nodes[:num_nodes_to_remove]]
#
#     # Eliminar los nodos más conectados de la red
#     grafototal.remove_nodes_from(nodes_to_remove)
#     grafodirigido.remove_nodes_from(nodes_to_remove)
#     grafototalsimple.remove_nodes_from(nodes_to_remove)
#     # Verificar el número de nodos restantes
#     print(f"Cantidad de nodos después de eliminar los más conectados: {len(grafototal.nodes())}")
#
#
#
# #Vuelve a mostrar la metacentralidad:
#
#     centralidad_buena_redux = pd.Series(nx.degree_centrality(grafototal))
#     bet_buena_redux = pd.Series(nx.betweenness_centrality(grafototal))
#     close_buena_redux = pd.Series(nx.closeness_centrality(grafototal))
#
#     centralidad_normalizada_redux = centralidad_buena_redux / centralidad_buena_redux.max()
#     bet_normalizada_redux = bet_buena_redux / bet_buena_redux.max()
#     close_normalizada_redux = close_buena_redux / close_buena_redux.max()
#
#     metacentralidad_redux = centralidad_normalizada_redux + bet_normalizada_redux + close_normalizada_redux
#     print("\n \n La metacentralidad tras eliminar los nodos es:")
#     print(metacentralidad_redux.sort_values(ascending=False))
# # ----------------------------------------------------------------

#

# #INFOMAP
#     # Paso 1: Mapear nodos a enteros
#     node2id = {node: idx for idx, node in enumerate(grafodirigido.nodes())}
#     id2node = {idx: node for node, idx in node2id.items()}
#
#     # Paso 2: Ejecutar Infomap 300 veces y guardar particiones
#     particiones = []
#
#     for i in range(100):
#         im = Infomap("--directed")  # Solo comunidades planas
#         for u, v in grafodirigido.edges():
#             im.add_link(node2id[u], node2id[v])
#         im.run()
#
#         particion = tuple(sorted((id2node[node.node_id], node.module_id) for node in im.nodes))
#         particiones.append(particion)
#
#     # Paso adicional: construir matriz de co-asignación
#     co_asignaciones = defaultdict(int)
#     total_iteraciones = 100
#
#     for particion in particiones:
#         comunidad_dict = {}
#         for nodo, comunidad in particion:
#             comunidad_dict.setdefault(comunidad, []).append(nodo)
#
#         for miembros in comunidad_dict.values():
#             for u, v in combinations(sorted(miembros), 2):
#                 co_asignaciones[(u, v)] += 1
#
#     # Paso: crear grafo de consenso (solo pares con co-asignación >= threshold)
#     threshold = 0.8  # 80%
#     min_repeticiones = int(total_iteraciones * threshold)
#
#     grafoconsenso = nx.Graph()
#     grafoconsenso.add_nodes_from(grafodirigido.nodes())
#
#     for (u, v), count in co_asignaciones.items():
#         if count >= min_repeticiones:
#             grafoconsenso.add_edge(u, v)
#
#     # Paso: detectar comunidades conectadas como consenso final
#     componentes = list(nx.connected_components(grafoconsenso))
#
#     print("\nSubcomunidades dentro de cada comunidad consenso:")
#
#     for i, comunidad in enumerate(componentes):
#         print(f"\nSubcomunidades en comunidad {i + 1}:")
#
#         # Subgrafo inducido
#         subgrafo = grafoconsenso.subgraph(comunidad)
#
#         # Mapear nodos a id para Infomap
#         sub_node2id = {node: idx for idx, node in enumerate(subgrafo.nodes())}
#         sub_id2node = {idx: node for node, idx in sub_node2id.items()}
#
#         # Ejecutar Infomap jerárquico en el subgrafo
#         im_sub = Infomap()  # jerárquico
#
#         for u, v in subgrafo.edges():
#             im_sub.add_link(sub_node2id[u], sub_node2id[v])
#         im_sub.run()
#
#         # Extraer subcomunidades
#         subcomunidades = defaultdict(list)
#         for node in im_sub.tree:
#             if node.is_leaf:
#                 ruta = node.path  # camino jerárquico
#                 nombre = sub_id2node[node.node_id]
#                 comunidad_sub = tuple(ruta[:-1])
#                 subcomunidades[comunidad_sub].append(nombre)
#
#         # Mostrar subcomunidades encontradas
#         for nivel, nodos in subcomunidades.items():
#             print(f"  Nivel {nivel}: {sorted(nodos)}")
#
#     # Leer Excel desde la ruta proporcionada
#     df_ids = pd.read_excel(r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\fisicamatesidentificador.xlsx")
#
#     # Usar la primera columna como índice (que contiene los nombres de nodos)
#     df_ids.set_index(df_ids.columns[0], inplace=True)
#
#     # Crear diccionario: nodo → identificador
#     diccionario_ids = df_ids["Identificador"].to_dict()
#
#     # Asignar comunidad a cada nodo
#     comunidad_por_nodo = {}
#     for i, comunidad in enumerate(componentes):  # componentes = lista de comunidades
#         for nodo in comunidad:
#             comunidad_por_nodo[nodo] = f"comunidad_{i + 1}"
#
#     # Añadir atributos a cada nodo en grafodirigido
#     for nodo in grafodirigido.nodes():
#         grafodirigido.nodes[nodo]["comunidad"] = comunidad_por_nodo.get(nodo, "Sin comunidad")
#         grafodirigido.nodes[nodo]["Identificador"] = diccionario_ids.get(nodo, "Sin identificador")
#
#     # Exportar grafo a GraphML para Cytoscape con el nombre final requerido
#     nx.write_graphml(grafodirigido, "Red_Infomap_FisicaMates_Eliminados.graphml")
#
#     # Crear diccionario nodo -> número comunidad (extraer número del string)
#     communities = {}
#     for nodo, com_str in comunidad_por_nodo.items():
#         try:
#             communities[nodo] = int(com_str.split("_")[1])
#         except:
#             communities[nodo] = -1  # Si no tiene comunidad asignada
#
#     # Contar enlaces entre comunidades en grafodirigido
#
#     edge_weights = defaultdict(int)
#     for u, v in grafodirigido.edges():
#         c_u = communities.get(u, -1)
#         c_v = communities.get(v, -1)
#         if c_u == -1 or c_v == -1:
#             continue  # Ignorar nodos sin comunidad
#         if c_u != c_v:
#             key = tuple(sorted((c_u, c_v)))
#             edge_weights[key] += 1
#
#     # Crear red de comunidades
#     G_communities = nx.Graph()
#     G_communities.add_nodes_from(set(communities.values()) - {-1})
#     for (c1, c2), weight in edge_weights.items():
#         G_communities.add_edge(c1, c2, weight=weight)
#
#     # Visualizar y guardar la red de comunidades
#
#
#     pos = nx.spring_layout(G_communities, seed=42)
#     weights = [G_communities[u][v]['weight'] for u, v in G_communities.edges()]
#     grosor = [w * 0.8 for w in weights]
#
#     plt.figure(figsize=(8, 6))
#     nx.draw_networkx_nodes(G_communities, pos, node_color='lightblue', node_size=1000)
#     nx.draw_networkx_edges(G_communities, pos, width=grosor, edge_color='gray')
#     nx.draw_networkx_labels(G_communities, pos, font_size=12, font_weight='bold')
#     plt.axis('off')
#     plt.title("Red de Comunidades")
#
#     ruta_guardado = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\Red_Comunidades_FisicaMates.png"
#     carpeta = os.path.dirname(ruta_guardado)
#     if carpeta and not os.path.exists(carpeta):
#         os.makedirs(carpeta)
#
#     plt.savefig(ruta_guardado, dpi=300)
#     #plt.show()
#     plt.close()
#     print(f"Red guardada en: {ruta_guardado}")
#
#





















































# Consensus clustering

#
#     #LOUVAIN
#     # 🔹 1️⃣ Crear carpeta de resultados
#     output_folder = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM"
#     os.makedirs(output_folder, exist_ok=True)
#
#     num_iterations = 500  # Número de iteraciones para el consenso
#
#     # 🔹 2️⃣ Mapeo de nodos
#     node_to_index = {node: i for i, node in enumerate(grafototalsimple.nodes())}
#     index_to_node = {i: node for node, i in node_to_index.items()}
#
#     # 🔹 3️⃣ Ejecutar Louvain varias veces para construir la matriz de co-ocurrencia
#     n = len(grafototalsimple.nodes())
#     co_occurrence_matrix = np.zeros((n, n))
#
#     for _ in range(num_iterations):
#         partition = community_louvain.best_partition(grafototalsimple)
#         for i in range(n):
#             for j in range(n):
#                 if partition[index_to_node[i]] == partition[index_to_node[j]]:
#                     co_occurrence_matrix[i, j] += 1
#
#     co_occurrence_matrix /= num_iterations
#
#     # 🔹 4️⃣ Consensus Clustering (clustering jerárquico)
#     distance_matrix = 1 - co_occurrence_matrix
#     linkage_matrix = linkage(squareform(distance_matrix), method='average')
#     threshold = 0.5
#     clusters = fcluster(linkage_matrix, threshold, criterion='distance')
#
#     # 🔹 5️⃣ Asignar comunidades finales
#     consensus_partition = {index_to_node[i]: clusters[i] for i in range(n)}
#
#     # # 🔹 6️⃣ Guardar la red completa con las comunidades detectadas
#     # plt.figure(figsize=(19.2, 10.8))
#     # pos = nx.kamada_kawai_layout(grafototalsimplesimple)
#     # colors = [consensus_partition[node] for node in grafototalsimplesimple.nodes()]
#     # nx.draw(grafototalsimplesimple, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1, node_size=500,
#     #         edge_color="gray")
#     # plt.title("Red Completa con Comunidades Detectadas", fontsize=14)
#     # plt.savefig(os.path.join(output_folder, "Fisica_Mates_red_completa.png"), dpi=300, bbox_inches='tight')
#     # plt.close()
#
#     # 🔹 7️⃣ Detectar y analizar subcomunidades
#     subcommunities = {}
#
#     for community_id in set(consensus_partition.values()):
#         subgraph_nodes = [node for node in consensus_partition if consensus_partition[node] == community_id]
#         subgraph = grafototalsimple.subgraph(subgraph_nodes)
#
#         # Consensus Clustering en la subcomunidad
#         if len(subgraph.nodes()) > 1:  # Evitar errores con comunidades de 1 nodo
#             num_subnodes = len(subgraph.nodes())
#             co_occurrence_sub = np.zeros((num_subnodes, num_subnodes))
#             sub_node_to_index = {node: i for i, node in enumerate(subgraph.nodes())}
#             sub_index_to_node = {i: node for node, i in sub_node_to_index.items()}
#
#             for _ in range(num_iterations):
#                 sub_partition = community_louvain.best_partition(subgraph)
#                 for i in range(num_subnodes):
#                     for j in range(num_subnodes):
#                         if sub_partition[sub_index_to_node[i]] == sub_partition[sub_index_to_node[j]]:
#                             co_occurrence_sub[i, j] += 1
#
#             co_occurrence_sub /= num_iterations
#             distance_sub = 1 - co_occurrence_sub
#             linkage_sub = linkage(squareform(distance_sub), method='average')
#             sub_clusters = fcluster(linkage_sub, threshold, criterion='distance')
#
#             sub_partition_final = {sub_index_to_node[i]: sub_clusters[i] for i in range(num_subnodes)}
#             subcommunities[community_id] = sub_partition_final
#
#             # # 🔹 8️⃣ Guardar imagen de la subcomunidad con sus sub-subcomunidades
#             # plt.figure(figsize=(19.2, 10.8))
#             # pos = nx.kamada_kawai_layout(subgraph)
#             # colors = [sub_partition_final[node] for node in subgraph.nodes()]
#             # nx.draw(subgraph, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1, node_size=500,
#             #         edge_color="gray")
#             # plt.title(f"Comunidad {community_id} - Subcomunidades", fontsize=14)
#             # plt.savefig(os.path.join(output_folder, f"Fisica_Mates_comunidad_{community_id}.png"), dpi=300, bbox_inches='tight')
#             # plt.close()
#
#     # 🔹 9️⃣ Imprimir las comunidades y subcomunidades detectadas
#     for community_id, sub_partition in subcommunities.items():
#         print(f"\n🔹 Comunidad {community_id}:")
#         sub_communities_sorted = {}
#
#         for node, sub_id in sub_partition.items():
#             if sub_id not in sub_communities_sorted:
#                 sub_communities_sorted[sub_id] = []
#             sub_communities_sorted[sub_id].append(node)
#
#         for sub_id, nodes in sub_communities_sorted.items():
#             print(f"  🔸 Subcomunidad {sub_id}: {sorted(nodes)}")
#
#         # 🔟 Añadir identificadores jerárquicos personalizados y nombre original
#
#         # Leer Excel con los nuevos identificadores
#         df_ids = pd.read_excel(r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\fisicamatesidentificador.xlsx")  # Cambia por tu ruta real
#         custom_ids = dict(zip(df_ids.iloc[:, 0], df_ids.iloc[:, 1]))  # nombre_original -> nuevo_id
#
#         # Guardar nombre original como atributo
#         original_names = {node: str(node) for node in grafototalsimple.nodes()}
#         nx.set_node_attributes(grafototalsimple, original_names, name="nombre_original")
#
#         # Guardar identificador personalizado como atributo
#         nx.set_node_attributes(grafototalsimple, custom_ids, name="identificador_personalizado")
#
#         # Crear atributos de comunidad y subcomunidad
#         comunidad_attr = {}
#         subcomunidad_attr = {}
#
#         for comunidad_id, sub_partition in subcommunities.items():
#             for node, sub_id in sub_partition.items():
#                 comunidad_attr[node] = comunidad_id
#                 subcomunidad_attr[node] = sub_id
#
#         nx.set_node_attributes(grafototalsimple, comunidad_attr, name="comunidad")
#         nx.set_node_attributes(grafototalsimple, subcomunidad_attr, name="subcomunidad")
#
#         # Renombrar nodos con los IDs personalizados
#         grafototalsimple_renombrado = nx.relabel_nodes(grafototalsimple, custom_ids)
#
#         # Exportar el grafo final con todos los atributos
#         #output_graphml = os.path.join(output_folder, "Red_Fisica_eliminada_final.graphml")
#         #nx.write_graphml(grafototalsimple_renombrado, output_graphml)
#
#
# #
# #Red de comunidades
#
#     # 🔹 1️⃣ Crear el grafo de comunidades
#     community_graph = nx.Graph()
#     communities = {}  # Diccionario para almacenar nodos en cada comunidad
#
#     # Agregar nodos de comunidades a partir del consensus_partition
#     for node, community_id in consensus_partition.items():
#         if community_id not in communities:
#             communities[community_id] = []
#         communities[community_id].append(node)
#
#     for community_id in communities.keys():
#         community_graph.add_node(community_id)  # Cada nodo representa una comunidad
#
#     # 🔹 2️⃣ Agregar enlaces entre comunidades si hay conexiones entre sus nodos en la red original
#     for u, v in grafototalsimple.edges():
#         comm_u = consensus_partition[u]
#         comm_v = consensus_partition[v]
#         if comm_u != comm_v:  # Solo conectar comunidades diferentes
#             if community_graph.has_edge(comm_u, comm_v):
#                 community_graph[comm_u][comm_v]['weight'] += 0.1  # Incrementar peso si ya existe
#             else:
#                 community_graph.add_edge(comm_u, comm_v, weight=0.1)  # Nuevo enlace
#
#     # 🔹 3️⃣ Mostrar qué nodos están en cada comunidad
#     print("\n📌 **Comunidades detectadas (Consensus Clustering):**")
#     for community_id, nodes in communities.items():
#         print(f"🔹 Comunidad {community_id}: {sorted(nodes)}")
#
#     # 🔹 4️⃣ Dibujar y guardar la red de comunidades
#     plt.figure(figsize=(19.2, 10.8))
#     pos = nx.spring_layout(community_graph)  # Diseño de nodos
#     weights = [community_graph[u][v]['weight'] for u, v in community_graph.edges()]  # Pesos de las aristas
#
#     nx.draw(community_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=weights,
#             node_size=2000, font_size=24)
#
#     plt.title("Red de comunidades Conjunta", fontsize=24)
#
#     # 🔹 Guardar imagen en la carpeta de resultados
#     output_folder = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM"
#     os.makedirs(output_folder, exist_ok=True)
#     plt.savefig(os.path.join(output_folder, "Red_de_comunidades_FisicaMates.png"), dpi=300, bbox_inches='tight')
#     plt.close()
#
#     plt.show()



# #Directed acyclic graphs. Utilizo solamente la red de enlaces dirigidos
#
#     #IDENTIFICADORES
#     #DIRECTED ACYCLIC GRAPH
#     ruta_excel_fisica = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\fisicaidentificador.xlsx"
#     ruta_excel_mates = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\matesidentificador.xlsx"
#     ruta_excel_fisicamates = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\fisicamatesidentificador.xlsx"
#
#     df = pd.read_excel(ruta_excel_fisica)
#     #print(df.head())
#
#     diccionario_ids = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
#
#
# # EMPEZAMOS
#
#     print("The number of edges is:", nx.number_of_edges(grafodirigido))
#
# # Relabel para asignar números a los nodos
# # Creamos un diccionario con los nodos originales y sus nuevos números
#     nodos_numerados = {node: str(i) for i, node in enumerate(grafototal.nodes)}
#
#     ciclo = 0
#     while not nx.is_directed_acyclic_graph(grafodirigido):
#         ciclo += 1
#         print(f"Is this graph a DAG? {nx.is_directed_acyclic_graph(grafodirigido)}")
#
#         print("El número de ciclos es:", len(list(nx.simple_cycles(grafodirigido))))
#
#         try:
#             cycle = nx.find_cycle(grafodirigido, orientation='original')
#             print("Cycle detected:", cycle)
#         except nx.NetworkXNoCycle:
#             print("No cycles detected, the graph is a DAG.")
#             break
#
#         if cycle:
#             for u, v, _ in cycle:
#                 if grafodirigido.has_edge(u, v):
#                     grafodirigido.remove_edge(u, v)
#                     print(f"Removed edge ({u}, {v}) to break the cycle.")
#                     break
#                 else:
#                     print(f"La arista ({u}, {v}) no existe en el grafo.")
#
#     print("Número de enlaces eliminados en la red normal:", ciclo)
#
# # Reemplazamos los nombres de los nodos
#     grafodirigido = nx.relabel_nodes(grafodirigido, diccionario_ids)
#
# # Mostrar los nodos con los números asignados
#     print("Nodos con nuevos IDs:", grafodirigido.nodes)
#
#     print(f"Is this graph a DAG? {nx.is_directed_acyclic_graph(grafodirigido)}")
#     topological_order = list(nx.topological_sort(grafodirigido))
#     print(f"Topological Order: {topological_order}")
#
# # 🔁 NUEVO: Calcular medida "básico-aplicado"
#     medida_basico_aplicado = {
#         node: grafodirigido.out_degree(node) - grafodirigido.in_degree(node)
#         for node in grafodirigido.nodes
#     }
#
# # 🔁 NUEVO: Normalizar valores y obtener colores
#     import matplotlib.colors as mcolors
#
#     valores = list(medida_basico_aplicado.values())
#     norm = mcolors.Normalize(vmin=min(valores), vmax=max(valores))
#     cmap = plt.cm.cividis  # azul-blanco-rojo
#     colores_nodos = [cmap(norm(medida_basico_aplicado[n])) for n in grafodirigido.nodes]
#
# # Crear figura y ejes
#     fig, ax = plt.subplots(figsize=(24, 14))
#
# # Calcular posiciones usando Graphviz
#     pos = graphviz_layout(grafodirigido, prog='dot')
#
# # Dibujar el grafo en los ejes
#     nx.draw(
#         grafodirigido, pos, with_labels=True, ax=ax,
#         node_color=colores_nodos, edge_color="gray",
#         node_size=2500, font_size=10, font_weight='bold', font_color='white',
#     )
#
# # Crear el ScalarMappable en el contexto del eje
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#
# # Añadir barra de color al eje
#     cbar = fig.colorbar(sm, ax=ax)
#     cbar.set_label('Medida "básico-aplicado" (out-degree - in-degree)', fontsize=12)
#
# # Título
#     plt.title("Jerarquía nodos red conjunta", fontsize=20, y=1.05)
#
# # Guardar imagen
#     carpeta = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\Texto_TFM\imagenes"
#     nombre_archivo = "jerarquia_red_fisica_color_epistemologico.png"
#     ruta_salida = os.path.join(carpeta, nombre_archivo)
#     plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
#
# # Mostrar el grafo
#     plt.show()
#
# # Imprimir número de aristas
#     print("The number of edges is:", nx.number_of_edges(grafodirigido))
#     print("Número de enlaces eliminados en la red normal:", ciclo)
#
#
#
#
# #RANDOMIZACIÓN DE REDES
#
#
#     # Obtener grados de entrada y salida de la red original
#     in_degrees = [d for _, d in grafodirigido.in_degree()]
#     out_degrees = [d for _, d in grafodirigido.out_degree()]
#
#     num_redes = 10000  # Número de redes aleatorias a generar
#     redes_random_dags = []
#     ciclos_eliminados_lista = []
#
#     for i in range(num_redes):
#         # Generar red aleatoria con mismo grado
#         g_conf = nx.directed_configuration_model(out_degrees, in_degrees, create_using=nx.DiGraph)
#
#         # Eliminar multiaristas y bucles
#         g_simple = nx.DiGraph(g_conf)
#         g_simple.remove_edges_from(nx.selfloop_edges(g_simple))
#
#         g_dag = g_simple.copy()
#         eliminados = 0
#
#         # Convertir en DAG eliminando ciclos
#         while not nx.is_directed_acyclic_graph(g_dag):
#             try:
#                 ciclo_detectado = nx.find_cycle(g_dag, orientation='original')
#             except nx.NetworkXNoCycle:
#                 break
#
#             for u, v, _ in ciclo_detectado:
#                 if g_dag.has_edge(u, v):
#                     g_dag.remove_edge(u, v)
#                     eliminados += 1
#                     break
#
#         redes_random_dags.append(g_dag)
#         ciclos_eliminados_lista.append(eliminados)
#
#     # Crear histograma con bins centrados en valores naturales (1, 2, 3, ...)
#     plt.figure(figsize=(10, 6))
#
#     # Definir valores únicos posibles (natural numbers)
#     min_val = min(ciclos_eliminados_lista)
#     max_val = max(ciclos_eliminados_lista)
#
#     # Centros de los bins en los enteros
#     centros = np.arange(min_val, max_val + 1)
#     # Crear bins de borde izquierdo/derecho para que el centro sea exactamente el entero
#     bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
#
#     # Dibujar histograma
#     plt.hist(ciclos_eliminados_lista, bins=bins, color='skyblue', edgecolor='black')
#
#     # Colocar ticks justo en los enteros
#     plt.xticks(centros)
#
#     # Línea vertical para la red original
#     plt.axvline(x=ciclo, color='red', linestyle='--', linewidth=2,
#                 label=f'Red original: {ciclo} enlaces eliminados')
#
#     plt.xlabel("Número de enlaces eliminados para obtener un DAG")
#     plt.ylabel("Frecuencia")
#     plt.title("Distribución de enlaces eliminados en redes aleatorias")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     # Guardar gráfico
#     carpeta = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\Texto_TFM\imagenes"
#     ruta_histograma = os.path.join(carpeta, "histograma_enlaces_eliminados_Fisica.png")
#     plt.savefig(ruta_histograma, dpi=300)
#
#     plt.show()
#
#     # Calcular p-valor (redes aleatorias con menos o igual enlaces eliminados que la real)
#     num_menores = sum(1 for x in ciclos_eliminados_lista if x <= ciclo)
#     p_valor = num_menores / len(ciclos_eliminados_lista)
#     print(f"P-valor aproximado (menos o igual ciclos que la red real): {p_valor:.3f}")
























if __name__ == "__main__":
    main()