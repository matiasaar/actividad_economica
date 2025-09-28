# data/loader.py

import os
import logging
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from tqdm import tqdm
from data.preprocessor import *
from data.get_data_bucket import *

import sys
 

# Agregar carpeta padre al path de búsqueda de módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_DIR, TEXT_DATA_FILENAME, ACTIVITY_CODES_FILENAME, SII_DATA_FILENAME

# ==========================
# Logging y constantes
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ==========================
# Funciones de carga de datos
# ==========================

def LoadTexts(path_to_text: str = TEXT_DATA_FILENAME) -> Tuple[List[str], List[str]]:
    """
    Carga textos y etiquetas desde un archivo TSV.
    Aplica GetUniqueTexts para asegurar entradas únicas.
    """
    ruta_guardado_textos = os.path.join(DATA_DIR, path_to_text)
    texts: List[str] = []
    labels: List[str] = []

    if os.path.exists(ruta_guardado_textos):
        with open(ruta_guardado_textos, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    label, text = line.strip().split('\t', 1)
                    texts.append(text)
                    labels.append(label)
                except ValueError:
                    logging.warning("Línea con formato incorrecto en %s: %s", path_to_text, line)
    else:
        logging.error("Archivo no encontrado: %s", ruta_guardado_textos)
        return [], []

    logging.info("Total de textos cargados desde %s: %d", path_to_text, len(texts))
    logging.info("Total de etiquetas cargadas desde %s: %d", path_to_text, len(labels))

    # Asegurarse de que los textos sean únicos
    textos_unicos, labels_unicos = GetUniqueTexts(texts, labels)
    return textos_unicos, labels_unicos


def load_activity_codes_data(filename: str = ACTIVITY_CODES_FILENAME) -> pd.DataFrame:
    """
    Carga el archivo Excel que contiene los mapeos de códigos de actividad a rubros.
    """
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        logging.info("Cargando tabla de códigos de actividad desde: %s", filepath)
        df = pd.read_excel(filepath)
        logging.info("Columnas en '%s': %s", filename, df.columns.tolist())
        return df
    else:
        logging.error("Archivo de códigos de actividad no encontrado: %s", filepath)
        return pd.DataFrame()


def load_sii_data_complete(filename: str = SII_DATA_FILENAME) -> pd.DataFrame:
    """
    Carga los datos del SII (v_sii_2.gzip) y realiza el filtrado inicial.
    """
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        logging.info("Cargando datos del SII desde: %s", filepath)
        df = pd.read_parquet(filepath, engine='pyarrow')
        logging.info("Shape original del SII: %s", df.shape)
        df = df[df['Rubro económico'] != 'Valor por Defecto']
        logging.info("Shape del SII después de filtrar 'Valor por Defecto': %s", df.shape)
        df.rename(columns={'RUT': 'rutnum'}, inplace=True)
        df['RUT'] = df['rutnum'].astype(str) + '-' + df['DV'].astype(str)
        return df
    else:
        logging.error("Archivo de datos del SII no encontrado: %s", filepath)
        return pd.DataFrame()


def load_data_and_preprocess(args: Any, ruts_to_process_ids: List[str]) -> Optional[Dict[str, Any]]:
    """
    Carga y preprocesa todos los datos necesarios, ya sea desde S3 o archivos locales.

    Args:
        args: Argumentos con flags como `new_bucket_data` y `solo_un_rubro`.
        ruts_to_process_ids: Lista de RUTs a procesar.

    Returns:
        Dict con 'rubros_por_rut' y '_rut_dict' si la carga es exitosa, o None en caso de error.
    """
    if args.new_bucket_data:
        logging.info("Cargando y preprocesando datos desde S3...")

        LAMBDA_URL = os.getenv("LAMBDA_URL")
        BUCKET_NAME = os.getenv("BUCKET_NAME")

        if not LAMBDA_URL or not BUCKET_NAME:
            logging.error("Las variables de entorno LAMBDA_URL y BUCKET_NAME deben estar definidas")
            return None

        try:
            creds = get_aws_auth(LAMBDA_URL, service="lambda")
            s3_client = create_s3_client(creds)
        except Exception as e:
            logging.error("Error al obtener las credenciales de AWS: %s", e)
            return None

        rut_dict_from_s3: Dict[str, Any] = {}
        processed_texts_s3: List[str] = []
        codes = load_activity_codes_data(ACTIVITY_CODES_FILENAME)
        sii = load_sii_data_complete(SII_DATA_FILENAME)

        ruts_to_process_ids_to_num = {
            int(i.replace('.', '').split('-')[0]): i
            for i in ruts_to_process_ids
        }

        for rut_sin_guion, rut_original in tqdm(ruts_to_process_ids_to_num.items(), desc="Procesando RUTs desde S3"):
            folder = f"portal-sii-xml/{rut_sin_guion}/"
            archivos = list_s3_files(s3_client, BUCKET_NAME, folder)

            textos_s3: List[str] = []
            for key in archivos:
                xml_str = read_s3_file(s3_client, BUCKET_NAME, key)
                if xml_str:
                    xml_dict = parse_xml_string(xml_str)
                    if xml_dict:
                        texto, _, _ = extract_fields(xml_dict)
                        textos_s3.append(texto)

            rut_dict_from_s3[rut_original] = {"emisor": textos_s3, "receptor": []}
            processed_texts_s3.extend(textos_s3)

        labels_from_texts = [txt.split(' ')[0].split(':')[1] if 'TipoDTE' in txt else None for txt in processed_texts_s3]
        labels_clean = [code.lstrip('0') if code else None for code in labels_from_texts]
        labels_map = map_codes_to_rubros(codes, labels_clean)
        ruts_em_s3, ruts_re_s3, giros_s3 = extract_ruts_and_giros_from_texts_codes(processed_texts_s3)
        rubros_por_rut_s3 = obtener_rubros_por_rut(sii, ruts_em_s3, ruts_re_s3, labels_map, giros_s3, args.solo_un_rubro)

        rut_dict_from_s3 = {str(rut): datos for rut, datos in rut_dict_from_s3.items() if rut in ruts_to_process_ids}
        rubros_por_rut_s3 = {str(rut): datos for rut, datos in rubros_por_rut_s3.items() if rut in ruts_to_process_ids}

        return {
            'rubros_por_rut': rubros_por_rut_s3,
            '_rut_dict': rut_dict_from_s3
        }

    else:
        logging.info("Cargando y preprocesando datos desde archivos locales...")
        all_texts, _ = LoadTexts(TEXT_DATA_FILENAME)
        codes = load_activity_codes_data(ACTIVITY_CODES_FILENAME)
        sii = load_sii_data_complete(SII_DATA_FILENAME)

        labels, texts = zip(*[i.split('\t', 1) if '\t' in i else (None, i) for i in all_texts])
        labels_clean = [code.split()[0].lstrip('0') if code else None for code in labels]
        labels_map = map_codes_to_rubros(codes, labels_clean)

        ruts_em, ruts_re, giros = extract_ruts_and_giros_from_texts_codes(texts)
        rut_dict = build_rut_text_dictionary(ruts_em, ruts_re, texts)
        rubros_por_rut = obtener_rubros_por_rut(sii, ruts_em, ruts_re, labels_map, giros, args.solo_un_rubro)

        rut_dict = {str(rut): datos for rut, datos in rut_dict.items() if rut in ruts_to_process_ids}
        rubros_por_rut = {str(rut): datos for rut, datos in rubros_por_rut.items() if rut in ruts_to_process_ids}

        return {
            'rubros_por_rut': rubros_por_rut,
            '_rut_dict': rut_dict
        }
