#data/rubros.py

from typing import List, Dict, Union
import pandas as pd
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def obtener_rubros_por_rut(
    df_rubros: pd.DataFrame,
    ruts_emisor: List[str],
    ruts_receptor: List[str],
    labels_code_to_rubro: List[Union[str, List[str]]],
    rubro_receptor: List[Union[str, List[str]]],
    solo_un_rubro: bool = True
) -> Dict[str, List[str]]:
    """
    Construye un diccionario de RUTs a rubros económicos.

    Args:
        df_rubros (pd.DataFrame): DataFrame con columnas ['RUT', 'Rubro económico', 'Año comercial'].
        ruts_emisor (List[str]): Lista de RUTs emisores.
        ruts_receptor (List[str]): Lista de RUTs receptores.
        labels_code_to_rubro (List[Union[str, List[str]]]): Rubros de los emisores.
        rubro_receptor (List[Union[str, List[str]]]): Rubros de los receptores.
        solo_un_rubro (bool): Si True, se mantiene solo un rubro por RUT (el más reciente).

    Returns:
        Dict[str, List[str]]: Diccionario {RUT: [rubros]}.
    """
    rut_dict_rubros: Dict[str, set] = defaultdict(set)

    # --- Procesar rubros desde el DataFrame del SII ---
    if solo_un_rubro:
        logging.info("Agrupando df_rubros por RUT, manteniendo solo un rubro por RUT (el más reciente).")
        df_rubros = df_rubros.sort_values("Año comercial", ascending=False)
        df_grouped = df_rubros.groupby("RUT").first()
        for rut, row in df_grouped.iterrows():
            rut_dict_rubros[rut].add(row["Rubro económico"])
    else:
        logging.info("Manteniendo todos los rubros del DataFrame sin filtrar.")
        for _, row in df_rubros.iterrows():
            rut_dict_rubros[row["RUT"]].add(row["Rubro económico"])

    # --- Añadir rubros desde etiquetas extraídas de textos ---
    for emisor, receptor, rubros_em, rubros_rec in zip(ruts_emisor, ruts_receptor, labels_code_to_rubro, rubro_receptor):
        # Rubros del emisor
        if isinstance(rubros_em, list):
            for r in rubros_em:
                rut_dict_rubros[emisor].add(r)
        else:
            rut_dict_rubros[emisor].add(rubros_em)
        # Rubros del receptor
        if isinstance(rubros_rec, list):
            for r in rubros_rec:
                rut_dict_rubros[receptor].add(r)
        else:
            rut_dict_rubros[receptor].add(rubros_rec)

    logging.info(f"Total RUTs con rubros asignados: {len(rut_dict_rubros)}")
    return {k: list(v) for k, v in rut_dict_rubros.items()}
