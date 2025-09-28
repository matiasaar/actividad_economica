# data/sii_parser.py

import pandas as pd
import logging
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def analyze_sii_rubros(v_sii_completa: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Analiza el DataFrame del SII para identificar RUTs con uno o varios rubros.

    Args:
        v_sii_completa (pd.DataFrame): DataFrame del SII con columnas 'RUT' y 'Rubro económico'.

    Returns:
        Tuple[pd.Series, pd.Series]: 
            - RUTs con un solo rubro
            - RUTs con más de un rubro
    """
    if v_sii_completa.empty:
        logging.warning("DataFrame del SII está vacío. No se puede realizar el análisis de rubros.")
        return pd.Series([], dtype=str), pd.Series([], dtype=str)

    # Contar número de rubros únicos por RUT
    count_rubro = v_sii_completa.groupby('RUT')['Rubro económico'].nunique()

    # Separar RUTs por cantidad de rubros
    ruts_solo_un_rubro = count_rubro[count_rubro == 1].reset_index().RUT
    ruts_mas_de_un_rubro = count_rubro[count_rubro != 1].reset_index().RUT

    # Comprobación interna de consistencia
    assert len(ruts_solo_un_rubro) + len(ruts_mas_de_un_rubro) == len(count_rubro), \
        'La suma de RUTs con uno y más de un rubro no coincide con el total.'

    logging.info(f"RUTs con un solo rubro: {len(ruts_solo_un_rubro)}")
    logging.info(f"RUTs con más de un rubro: {len(ruts_mas_de_un_rubro)}")

    return ruts_solo_un_rubro, ruts_mas_de_un_rubro
