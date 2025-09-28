#data/extract.py

import re
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def extraer_info_concatenada(texto: str) -> Dict[str, str | List[str]]:
    """
    Extrae información relevante de un texto concatenado de documentos SII.

    Args:
        texto (str): Texto plano donde se buscarán patrones específicos.

    Returns:
        Dict[str, Union[str, List[str]]]: Diccionario con campos:
            - emisor
            - receptor
            - giro_comprador
            - consumidor_final
            - productos
    """
    # --- Buscar patrones específicos con expresiones regulares ---
    emisor = re.search(r'razon social del emisor\s+(.+?)\s+rut del receptor', texto, re.IGNORECASE)
    receptor = re.search(r'razon social del receptor\s+(.+?)\s+giro del receptor', texto, re.IGNORECASE)
    giro = re.search(r'giro del receptor\s+(.+?)\s+monto neto', texto, re.IGNORECASE)
    consumidor = re.search(r'venta a consumidor final\s+(\w+)', texto, re.IGNORECASE)
    productos = re.findall(r'nombre del producto\s+(.+?)\s+cantidad del producto', texto, re.IGNORECASE)

    # --- Construir el diccionario de salida ---
    resultado: Dict[str, str | List[str]] = {
        "emisor": emisor.group(1).strip() if emisor else "Desconocido",
        "receptor": receptor.group(1).strip() if receptor else "Desconocido",
        "giro_comprador": giro.group(1).strip() if giro else "No especificado",
        "consumidor_final": consumidor.group(1).capitalize() if consumidor else "No especificado",
        "productos": [p.strip() for p in productos] if productos else []
    }

    # --- Debug opcional ---
    logging.debug(f"Extraído info: {resultado}")

    return resultado
