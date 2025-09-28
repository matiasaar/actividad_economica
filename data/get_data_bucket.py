import os
import json
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

import boto3
from botocore.exceptions import ClientError
import requests
from requests_aws4auth import AWS4Auth

# ==========================
# Configuración Logging y dotenv
# ==========================
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
LAMBDA_URL: str = os.getenv("LAMBDA_URL", "")


# ==========================
# S3 Helpers
# ==========================

def get_aws_auth(lambda_url: str, service: str = "lambda") -> Dict[str, Any]:
    """
    Obtiene credenciales temporales llamando a un endpoint Lambda protegido con AWS SigV4.

    Args:
        lambda_url (str): URL del Lambda.
        service (str): Servicio AWS a firmar (por defecto 'lambda').

    Returns:
        dict: Respuesta JSON del Lambda con credenciales temporales.
    """
    load_dotenv()
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise ValueError("No se encontraron credenciales AWS en el archivo .env")

    aws_auth = AWS4Auth(
        access_key,
        secret_key,
        REGION,
        service,
        session_token=session_token
    )

    response = requests.get(lambda_url, auth=aws_auth)
    response.raise_for_status()
    logging.info("Credenciales AWS obtenidas correctamente.")
    return response.json()


def create_s3_client(creds: Dict[str, str]) -> boto3.client:
    """Crea un cliente boto3 S3 autenticado con credenciales temporales."""
    return boto3.client(
        "s3",
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=REGION
    )


def list_s3_files(s3_client: boto3.client, bucket: str, folder: str) -> List[str]:
    """Lista archivos dentro de un folder/prefix en S3."""
    if not folder.endswith("/"):
        folder += "/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        archivos: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=folder):
            if "Contents" in page:
                archivos.extend(obj["Key"] for obj in page["Contents"])
        return archivos
    except ClientError as e:
        logging.error("Error AWS al listar folder '%s': %s", folder, e)
        return []


def read_s3_file(s3_client: boto3.client, bucket: str, file_key: str) -> Optional[str]:
    """Lee un archivo desde S3 y devuelve su contenido como string."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        body = response["Body"].read()
        for encoding in ["utf-8", "latin1"]:
            try:
                return body.decode(encoding)
            except UnicodeDecodeError:
                continue
        logging.warning("No se pudo decodificar el archivo: %s", file_key)
        return None
    except ClientError as e:
        logging.error("Error AWS al leer '%s': %s", file_key, e)
        return None


# ==========================
# XML Helpers
# ==========================

def xml_to_dict(elem: ET.Element) -> Dict[str, Any]:
    """Convierte un nodo XML en un diccionario recursivo."""
    d: Dict[str, Any] = {elem.tag: {} if elem.attrib else None}
    children = list(elem)
    if children:
        dd: Dict[str, Any] = {}
        for dc in map(xml_to_dict, children):
            for key, value in dc.items():
                if key in dd:
                    if not isinstance(dd[key], list):
                        dd[key] = [dd[key]]
                    dd[key].append(value)
                else:
                    dd[key] = value
        d = {elem.tag: dd}
    if elem.text:
        text = elem.text.strip()
        if children or elem.attrib:
            if text:
                d[elem.tag]["text"] = text
        else:
            d[elem.tag] = text
    return d


def parse_xml_string(xml_str: str) -> Optional[Dict[str, Any]]:
    """Parses un XML string en dict."""
    try:
        root = ET.fromstring(xml_str)
        return xml_to_dict(root)
    except ET.ParseError as e:
        logging.error("Error al parsear XML: %s", e)
        return None


def extract_fields(xml_dict: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extrae campos claves del XML como texto y etiquetas.

    Returns:
        texto (str): Texto concatenado de campos y detalle.
        giro_emisor (str): Giro del emisor.
        acteco (str): Código económico (Acteco).
    """
    try:
        doc = xml_dict["SetDTE"]["DTE"]["Documento"]
        encabezado = doc["Encabezado"]
        iddoc = encabezado["IdDoc"]
        emisor = encabezado["Emisor"]
        receptor = encabezado["Receptor"]
        totales = encabezado["Totales"]
        detalle = doc.get("Detalle", [])

        detalles = detalle if isinstance(detalle, list) else [detalle]
        detalles_texto = [
            " ".join(
                f"{k}:{str(det.get(k, '')).strip()}"
                for k in ["NroLinDet", "NmbItem", "QtyItem", "UnmdItem", "PrcItem", "MontoItem"]
                if k in det
            )
            for det in detalles
        ]

        giro_emisor = emisor.get("GiroEmis", "").strip()
        acteco = emisor.get("Acteco", "")
        if isinstance(acteco, list):
            acteco = " ".join(str(x).strip() for x in acteco)
        elif isinstance(acteco, str):
            acteco = acteco.strip()
        else:
            acteco = str(acteco)

        campos = {
            "TipoDTE": iddoc.get("TipoDTE", ""),
            "FchEmis": iddoc.get("FchEmis", ""),
            "TpoTranCompra": iddoc.get("TpoTranCompra", ""),
            "TpoTranVenta": iddoc.get("TpoTranVenta", ""),
            "FmaPago": iddoc.get("FmaPago", ""),
            "RUTEmisor": emisor.get("RUTEmisor", ""),
            "RznSocEmisor": emisor.get("RznSoc", ""),
            "RUTRecep": receptor.get("RUTRecep", ""),
            "RznSocRecep": receptor.get("RznSocRecep", ""),
            "GiroRecep": receptor.get("GiroRecep", ""),
            "MntNeto": totales.get("MntNeto", ""),
            "TasaIVA": totales.get("TasaIVA", ""),
            "IVA": totales.get("IVA", ""),
            "MntTotal": totales.get("MntTotal", ""),
            "B2C": str(
                1 if (int(emisor.get("RUTEmisor", "0").split("-")[0]) >= 50e6
                      and int(receptor.get("RUTRecep", "0").split("-")[0]) < 50e6)
                else 0
            ),
        }

        partes: List[str] = []
        for k, v in campos.items():
            if v:
                clean_v = str(v).strip().replace("\n", " ")
                partes.append(f"{k}:{clean_v}")

        texto = " ".join(partes) + " " + " ".join(detalles_texto)
        return texto, giro_emisor, acteco

    except KeyError as e:
        logging.error("Campo faltante en XML: %s", e)
        return "", "", ""


# ==========================
# Procesamiento en lote
# ==========================

def procesar_rut(s3_client: boto3.client, rut: str) -> List[Dict[str, Any]]:
    """
    Procesa todos los XML de un RUT, devolviendo lista de dicts con textos y etiquetas.
    """
    folder = f"portal-sii-xml/{rut}/"
    archivos = list_s3_files(s3_client, BUCKET_NAME, folder)

    resultados: List[Dict[str, Any]] = []
    for key in archivos:
        xml_str = read_s3_file(s3_client, BUCKET_NAME, key)
        if not xml_str:
            continue
        xml_dict = parse_xml_string(xml_str)
        if not xml_dict:
            continue
        texto, giro_emisor, acteco = extract_fields(xml_dict)
        resultados.append({"file": key, "texto": texto, "giro_emisor": giro_emisor, "acteco": acteco})
    logging.info("Procesados %d archivos de RUT %s", len(resultados), rut)
    return resultados


# ==========================
# Main
# ==========================

def main() -> None:
    """Función principal para cargar datos desde S3 y procesar un RUT."""
    parser = argparse.ArgumentParser(description="Carga y procesa datos XML desde bucket S3")
    parser.add_argument("--rut", type=str, required=True, help="RUT objetivo (sin guion)")
    parser.add_argument("--max_docs", type=int, default=10, help="Max. docs a procesar por RUT.")
    parser.add_argument("--solo_un_rubro", action="store_true", help="Si se usa solo un rubro SII.")
    parser.add_argument("--new_bucket_data", action="store_true", help="Si se buscan datos directamente en S3.")
    args = parser.parse_args()

    creds = get_aws_auth(LAMBDA_URL, service="lambda")
    logging.info("Credenciales AWS recibidas.")

    s3_client = create_s3_client(creds)
    resultados = procesar_rut(s3_client, args.rut)

    for r in resultados[:2]:  # solo muestro 2 para debug
        logging.info(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
