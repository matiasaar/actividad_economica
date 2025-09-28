#data/preprocessor.py

import re
import unicodedata
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv # <--- NUEVA IMPORTACIÓN

# --- Cargar variables de entorno desde el archivo .env ---
load_dotenv()

# --- Importaciones de tu proyecto ---
from config import (
    TEXT_DATA_FILENAME, ACTIVITY_CODES_FILENAME, SII_DATA_FILENAME,
    REEMPLAZOS_LEGIBLES
)
#from data.loader import LoadTexts, load_activity_codes_data, load_sii_data_complete


def GetUniqueTexts(texts: list, labels: list):
    """
    Asegura que textos y labels tienen la misma longitud y retorna pares texto-etiqueta únicos.
    Mantiene la primera aparición de cada texto único.
    """
    assert len(texts) == len(labels), "Las listas no tienen la misma longitud."

    texto_label_unicos = {}
    for texto, label in zip(texts, labels):
        if texto not in texto_label_unicos:
            texto_label_unicos[texto] = label

    textos_unicos = list(texto_label_unicos.keys())
    labels_unicos = list(texto_label_unicos.values())

    print(f"Originales: {len(texts)}")
    print(f"Únicos: {len(textos_unicos)}")

    return textos_unicos, labels_unicos

def texto_legible_y_anonimo(texto: str, anonimizar_giro_receptor: bool = True):
    """
    Limpia, normaliza y anonimiza entidades específicas en el texto de entrada.
    Retorna el texto procesado y los RUTs extraídos.
    """
    texto = texto.lower()

    # Reemplazar separadores
    texto = texto.replace(":", " ").replace("/", " ").replace("_", " ")
    # Mantener guiones como en RUTs, según la lógica original. No se necesita cambio para '-' aquí.

    # Eliminar acentos
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")

    palabras = texto.split()
    resultado = []
    rut_emisor = None
    rut_receptor = None
    skip_next = False

    for i, palabra in enumerate(palabras):
        if skip_next:
            skip_next = False
            continue

        # Procesar RUT Emisor
        if palabra == "rutemisor" and i + 1 < len(palabras):
            rut_emisor = palabras[i + 1]
            resultado.append("rut del emisor <RUT_EMISOR>")
            skip_next = True
        # Procesar RUT Receptor
        elif palabra == "rutrecep" and i + 1 < len(palabras):
            rut_receptor = palabras[i + 1]
            resultado.append("rut del receptor <RUT_RECEPTOR>")
            skip_next = True
        # Procesar Giro del receptor
        elif palabra == "girorecep" and i + 1 < len(palabras):
            if anonimizar_giro_receptor:
                resultado.append("giro del receptor <GIRO_RECEPTOR>")
            else:
                resultado.append("giro del receptor")
                resultado.append(palabras[i + 1]) # Mantener el giro original
            skip_next = True
        # Procesar B2C dummy
        elif palabra == "b2c" and i + 1 < len(palabras):
            valor = palabras[i + 1]
            if valor == "1":
                resultado.append("venta a consumidor final si")
            elif valor == "0":
                resultado.append("venta a consumidor final no")
            skip_next = True
        # Aplicar reemplazos comunes
        elif palabra in REEMPLAZOS_LEGIBLES:
            resultado.append(REEMPLAZOS_LEGIBLES[palabra])
        else:
            resultado.append(palabra)

    texto_final = " ".join(resultado)
    #return texto_final, rut_emisor, rut_receptor #esto cambia en relacion al notebook
    return texto_final 

def extraer_info_concatenada(texto: str) -> str:
    """
    Extrae y concatena información relevante de un texto procesado
    para usar en prompts o visualización.
    """
    # Buscar fragmentos relevantes usando regex
    emisor = re.search(r'razon social del emisor\s+(.+?)\s+rut del receptor', texto, re.IGNORECASE)
    receptor = re.search(r'razon social del receptor\s+(.+?)\s+giro del receptor', texto, re.IGNORECASE)
    giro = re.search(r'giro del receptor\s+(.+?)\s+monto neto', texto, re.IGNORECASE)
    consumidor_final = re.search(r'venta a consumidor final\s+(\w+)', texto, re.IGNORECASE)
    productos = re.findall(r'nombre del producto\s+(.+?)\s+cantidad del producto', texto, re.IGNORECASE)

    # Extraer datos o usar valores por defecto
    nombre_vendedor = emisor.group(1).strip() if emisor else 'Desconocido'
    nombre_comprador = receptor.group(1).strip() if receptor else 'Desconocido'
    giro_comprador = giro.group(1).strip() if giro else 'No especificado'
    venta_consumidor = consumidor_final.group(1).strip().capitalize() if consumidor_final else 'No especificado'
    lista_productos = ', '.join(productos) if productos else 'Sin productos detectados'

    # Construir texto final
    resultado = (
        f"Nombre del vendedor: {nombre_vendedor}\n"
        f"Nombre del comprador: {nombre_comprador}\n"
        f"Giro del comprador: {giro_comprador}\n"
        f"Venta a consumidor final: {venta_consumidor}\n"
        f"Productos vendidos: {lista_productos}"
    )
    return resultado


def map_codes_to_rubros(tabla_codigo_to_rubro: pd.DataFrame, label_codes: list) -> list:
    """
    Mapea códigos de actividad numéricos a su 'Rubro' correspondiente desde un DataFrame.
    Maneja mapeos faltantes retornando 'SIN RUBRO'.
    """
    rubros = []
    for code in label_codes:
        # Asumiendo que `code` es una cadena que podría contener múltiples códigos o ser el cleaned_code[0]
        # del procesamiento anterior. Asegurémonos de que se trate como un solo entero para la búsqueda.
        try:
            int_code = int(code) # Convertir a int para la búsqueda en DataFrame
            # Filtrar DataFrame para el código específico
            
            filtered_df = tabla_codigo_to_rubro[tabla_codigo_to_rubro['Codigo'] == int_code]
            if not filtered_df.empty:
                # Obtener rubros únicos y tomar el primero
                rubros.append(filtered_df['Rubro'].unique()[0])
            else:
                rubros.append('SIN RUBRO')
        except ValueError: # Manejar casos donde el código podría no ser un entero
            rubros.append('SIN RUBRO')
    return rubros

def extract_ruts_and_giros_from_texts_codes(texts_codes: list):
    """
    Extrae RUTs (emisor, receptor) y el giro del receptor de una lista de textos procesados.
    """
    ruts_emisor = np.array([
        re.search(r'RUTEmisor:([0-9\-Kk]+)', text).group(1)
        if re.search(r'RUTEmisor:([0-9\-Kk]+)', text) else None
        for text in texts_codes
    ])
    ruts_receptor = np.array([
        re.search(r'RUTRecep:([0-9\-Kk]+)', text).group(1)
        if re.search(r'RUTRecep:([0-9\-Kk]+)', text) else None
        for text in texts_codes
    ])
    rubro_receptor = np.array([
        re.search(r'GiroRecep:(.*?)(?=\s\w+:|$)', text).group(1).strip()
        if re.search(r'GiroRecep:(.*?)(?=\s\w+:|$)', text) else None
        for text in texts_codes
    ])
    return ruts_emisor, ruts_receptor, rubro_receptor

def build_rut_text_dictionary(ruts_emisor: list, ruts_receptor: list, textos: list) -> dict:
    """
    Construye un diccionario que mapea los RUTs a listas de textos donde aparecen como emisor o receptor.
    """
    rut_dict = defaultdict(lambda: {"emisor": [], "receptor": []})
    for emisor, receptor, texto in zip(ruts_emisor, ruts_receptor, textos):
        if emisor:
            rut_dict[emisor]["emisor"].append(texto)
        if receptor:
            rut_dict[receptor]["receptor"].append(texto)
    return dict(rut_dict) # Convertir a dict regular para inmutabilidad si se prefiere


def obtener_rubros_por_rut(df_rubros: pd.DataFrame, ruts_emisor: list, ruts_receptor: list,
                          labels_code_to_rubro: list, rubro_receptor: list,
                          solo_un_rubro: bool = True) -> dict:
    """
    Construye un diccionario de rubros únicos asociados a cada RUT.
    Combina rubros de datos históricos y análisis de texto.
    """
    rut_dict_rubros = defaultdict(set)

    # Parte 1: Desde el DataFrame (datos históricos del SII)
    if solo_un_rubro:
        # Obtener el rubro más reciente para cada RUT
        df_sorted = df_rubros.sort_values(by="Año comercial", ascending=False)
        df_grouped = df_sorted.groupby("RUT").first().reset_index()
        for _, row in df_grouped.iterrows():
            if pd.notna(row["RUT"]) and pd.notna(row["Rubro económico"]):
                rut_dict_rubros[row["RUT"]].add(str(row["Rubro económico"]))
    else:
        # Obtener todos los rubros para cada RUT
        for _, row in df_rubros.iterrows():
            if pd.notna(row["RUT"]) and pd.notna(row["Rubro económico"]):
                rut_dict_rubros[row["RUT"]].add(str(row["Rubro económico"]))

    # Parte 2: Desde los textos parseados
    for emisor, receptor, rubros_emisor_list, rubro_recep_text in zip(
        ruts_emisor, ruts_receptor, labels_code_to_rubro, rubro_receptor
    ):
        if emisor and rubros_emisor_list:
            if isinstance(rubros_emisor_list, (list, np.ndarray)):
                for r in rubros_emisor_list:
                    if r: rut_dict_rubros[emisor].add(str(r))
            else: # Asumiendo que es una sola cadena
                if rubros_emisor_list: rut_dict_rubros[emisor].add(rubros_emisor_list)

        if receptor and rubro_recep_text:
            if isinstance(rubro_recep_text, (list, np.ndarray)):
                for r in rubro_recep_text:
                    if r: rut_dict_rubros[receptor].add(str(r))
            else: # Asumiendo que es una sola cadena
                if rubro_recep_text: rut_dict_rubros[receptor].add(str(rubro_recep_text))

    # Convertir sets a listas para mejor serialización/consistencia
    final_rut_rubros = {str(rut): list(rubros) for rut, rubros in rut_dict_rubros.items()}
    return final_rut_rubros


