# utils/helpers.py

import os
import pickle
import zipfile
import json
import re
import random
import logging
from datetime import datetime
from typing import Any, List, Dict, Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def guardar_pickle(data: Any, filename: str, directory: str) -> None:
    """
    Guarda datos en un archivo pickle en el directorio especificado.
    Asegura que el directorio exista.

    Args:
        data: Datos a guardar.
        filename: Nombre del archivo pickle.
        directory: Directorio donde se guardará el archivo.
    """
    filepath = os.path.join(directory, filename)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Datos guardados exitosamente en {filepath}")
    except Exception as e:
        logging.error(f"Error al guardar datos en {filepath}: {e}")


def cargar_pickle(filename: str, directory: str) -> Optional[Any]:
    """
    Carga datos de un archivo pickle en el directorio especificado.

    Args:
        filename: Nombre del archivo pickle.
        directory: Directorio donde se encuentra el archivo.

    Returns:
        Los datos cargados o None si ocurre un error.
    """
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Datos cargados exitosamente desde {filepath}")
            return data
        except Exception as e:
            logging.error(f"Error al cargar datos desde {filepath}: {e}")
            return None
    else:
        logging.error(f"Archivo no encontrado: {filepath}")
        return None


def load_ruts_from_file(filepath: str) -> List[str]:
    """
    Carga una lista de RUTs desde un archivo de texto (uno por línea).

    Args:
        filepath: Ruta del archivo.

    Returns:
        Lista de RUTs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ruts = [line.strip() for line in f if line.strip()]
        logging.info(f"Se cargaron {len(ruts)} RUTs desde '{filepath}'")
        return ruts
    except FileNotFoundError:
        logging.error(f"El archivo de RUTs '{filepath}' no fue encontrado.")
        return []


def cargar_datos_desde_zip(zip_filepath: str, nombres_a_cargar: List[str]) -> List[Any]:
    """
    Carga archivos .pkl específicos desde un archivo ZIP.

    Args:
        zip_filepath: Ruta al archivo ZIP.
        nombres_a_cargar: Lista de RUTs cuyos .pkl deben cargarse.

    Returns:
        Lista de objetos deserializados desde los archivos .pkl seleccionados.
    """
    datos: List[Any] = []
    logging.info(f"Cargando archivos .pkl desde el ZIP '{zip_filepath}'...")

    if not os.path.exists(zip_filepath):
        logging.error(f"El archivo ZIP '{zip_filepath}' no existe.")
        return []

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            pkl_files = [f for f in zf.namelist() if f.endswith(".pkl")]
            logging.info(f"El ZIP contiene {len(pkl_files)} archivos .pkl.")

            archivos_filtrados = [
                f for f in pkl_files
                if os.path.basename(f).removeprefix("salida_rubro_").removesuffix(".pkl").lstrip('0') in nombres_a_cargar
            ]

            if not archivos_filtrados:
                logging.warning("No se encontraron los archivos solicitados en el ZIP.")
                return []

            for filename in archivos_filtrados:
                try:
                    with zf.open(filename) as f:
                        datos.append(pickle.load(f))
                        logging.info(f"Archivo cargado: {filename}")
                except Exception as e:
                    logging.warning(f"No se pudo cargar el archivo {filename}. Error: {e}")

    except zipfile.BadZipFile:
        logging.error(f"El archivo '{zip_filepath}' no es un ZIP válido.")
        return []

    logging.info(f"Se cargaron {len(datos)} archivos .pkl desde el ZIP.")
    return datos


def extraer_contenido_entre_llaves(text: str) -> Dict:
    """
    Extrae contenido JSON encerrado entre llaves de una cadena.
    Asume que el contenido es un objeto JSON válido.

    Args:
        text: Texto que contiene el JSON.

    Returns:
        Diccionario con los datos extraídos o vacío si falla.
    """
    try:
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = text[start_index: end_index + 1]
            return json.loads(json_str)
        else:
            logging.warning("No se encontró un objeto JSON entre llaves.")
            return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error al decodificar JSON: {e}")
        return {}
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado durante la extracción de JSON: {e}")
        return {}


def _extraer_fecha(documento_str: str) -> Optional[datetime]:
    """
    Función auxiliar para extraer la fecha de un string de documento.

    Args:
        documento_str: Cadena que contiene la fecha en formato 'FchEmis:YYYY-MM-DD'.

    Returns:
        Objeto datetime o None si no se encuentra la fecha.
    """
    match = re.search(r"FchEmis:(\d{4}-\d{2}-\d{2})", documento_str)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    return None


def samplear_documentos_por_rut(
    rut_dict: Dict[str, Dict[str, List[str]]],
    ruts_a_procesar: List[str],
    metodo: str,
    n_muestras: int
) -> Dict[str, Dict[str, List[str]]]:
    """
    Realiza un muestreo de los documentos de emisor para una lista de RUTs.

    Args:
        rut_dict: Diccionario que contiene los datos, ej: {'RUT1': {'emisor': [...]}}.
        ruts_a_procesar: Lista de RUTs sobre los que se aplicará el muestreo.
        metodo: Método de muestreo ('aleatorio', 'recientes', 'antiguos', 'estratificado').
        n_muestras: Número de documentos a seleccionar.

    Returns:
        Diccionario con la misma estructura, pero con la lista de documentos 'emisor' muestreada.
    """
    resultado_muestreado: Dict[str, Dict[str, List[str]]] = {}

    for rut in ruts_a_procesar:
        if rut not in rut_dict or 'emisor' not in rut_dict[rut]:
            continue

        documentos_originales = rut_dict[rut]['emisor']

        if len(documentos_originales) <= n_muestras:
            resultado_muestreado[rut] = rut_dict[rut].copy()
            continue

        documentos_con_fecha = [(doc, _extraer_fecha(doc)) for doc in documentos_originales if _extraer_fecha(doc)]

        if not documentos_con_fecha:
            resultado_muestreado[rut] = rut_dict[rut].copy()
            resultado_muestreado[rut]['emisor'] = random.sample(documentos_originales, n_muestras)
            continue

        documentos_seleccionados: List[str] = []

        if metodo == 'aleatorio':
            seleccion = random.sample(documentos_con_fecha, n_muestras)
            documentos_seleccionados = [doc for doc, _ in seleccion]

        elif metodo == 'recientes':
            documentos_ordenados = sorted(documentos_con_fecha, key=lambda x: x[1], reverse=True)
            documentos_seleccionados = [doc for doc, _ in documentos_ordenados[:n_muestras]]

        elif metodo == 'antiguos':
            documentos_ordenados = sorted(documentos_con_fecha, key=lambda x: x[1])
            documentos_seleccionados = [doc for doc, _ in documentos_ordenados[:n_muestras]]

        elif metodo == 'estratificado':
            df_docs = pd.DataFrame(documentos_con_fecha, columns=['documento', 'fecha'])
            df_docs['estrato_tiempo'] = pd.cut(df_docs['fecha'], bins=n_muestras, labels=False)
            muestras_estratificadas = df_docs.groupby('estrato_tiempo').apply(lambda x: x.sample(1))
            documentos_seleccionados = muestras_estratificadas['documento'].tolist()

        else:
            raise ValueError(f"Método '{metodo}' no reconocido. Use 'aleatorio', 'recientes', 'antiguos' o 'estratificado'.")

        resultado_muestreado[rut] = rut_dict[rut].copy()
        resultado_muestreado[rut]['emisor'] = documentos_seleccionados

    return resultado_muestreado




def OnlyAnswer(texts_: List[str]) -> List[str]:
    """
    Elimina etiquetas de pensamiento del LLM y artefactos del prompt de una lista de textos.

    Args:
        texts_ (List[str]): Lista de textos con posibles etiquetas <think> y artefactos de prompt.

    Returns:
        List[str]: Lista de textos limpios, sin etiquetas de pensamiento ni artefactos.
    """
    # Usando regex para eliminar bloques <think>...</think> y "Texto corregido:\n"
    # re.DOTALL hace que '.' también coincida con saltos de línea
    texts_solo_respuesta = [
        re.sub(r"<think>.*?</think>\s*|Texto corregido:\n", "", i_, flags=re.DOTALL).strip()
        for i_ in texts_
    ]
    return texts_solo_respuesta