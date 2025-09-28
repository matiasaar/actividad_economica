#api_model.py

# --- 1. IMPORTACIONES Y CONFIGURACIÓN ---
import os
import json
import requests
from tqdm import tqdm
import argparse
import re
import random
from datetime import datetime
import pandas as pd

from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Dict, Any, Tuple
import asyncio
import aiohttp
import sys
import logging

# --- Importaciones del proyecto ---
from config import (
    TEXT_DATA_FILENAME, ACTIVITY_CODES_FILENAME, SII_DATA_FILENAME,
    RESULTS_DIR, RESUMEN_RUBROS_ADICIONALES,
    LLM_MODEL_NAME_API, LLM_TEMPERATURE, INNER_WORKERS, OUTER_WORKERS,
    OLLAMA_BASE_URL, CLASSIFICATION_RESULTS_DIR,URL_DEEP,URL_GPT
)

from data.loader import LoadTexts, load_activity_codes_data, load_sii_data_complete, load_data_and_preprocess
from data.preprocessor import (
    texto_legible_y_anonimo, extraer_info_concatenada 
)
from llm.prompts import (
    generar_prompt_completar_texto, generar_prompt_clasificacion, generar_prompt2
)
 
from openai import OpenAI
from utils.helpers import *
from dotenv import load_dotenv
# from data.get_data_bucket import *

# --- Configuración de logs ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

# --- Claves de API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEP_API_KEY = os.getenv("DEEP_API_KEY")

 
# --- Función para llamada sincrónica a LLM ---
def call_llm(prompt: str, model: str, temp: float, api_key: str, base_url: str) -> str:
    """Llama a la API de OpenAI/DeepSeek (sincrónica) y devuelve SOLO el texto de salida."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


# --- Función para llamada asíncrona a LLM ---
async def _async_call_llm(
    session: aiohttp.ClientSession, prompt: str, model: str, temp: float,
    api_key: str, base_url: str, semaphore: asyncio.Semaphore
) -> str:
    """
    Realiza una única llamada a la API de OpenAI/DeepSeek de forma asíncrona.
    Usa semáforo para limitar la concurrencia.
    """
    url = f"{base_url}/chat/completions"   #   ruta final para usar un solo nombre de url
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp
    }

    async with semaphore:
        try:
            async with session.post(url, json=payload, headers=headers, timeout=180) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"].strip()
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
            logging.error(f"Error en la llamada para el prompt '{prompt[:30]}...': {e}")
            return f"Error: {e}"


# --- Orquestador principal (nivel RUT) ---
async def ejecutar_prueba_async(
    ruts: List[str], model: str, temp: float, max_docs: int,
    args, solo_un_rubro: bool = False, new_bucket_data: bool = False
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Orquesta el proceso de carga, muestreo y llamada a la API de forma asíncrona.
    Cada RUT se procesa como una unidad independiente (nivel RUT).
    """
    logging.info(f"Iniciando prueba para {len(ruts)} RUT(s).")
    os.makedirs(args.output_dir, exist_ok=True)

    # Cargar y muestrear datos
    all_data = load_data_and_preprocess(args, ruts)
    all_data["_rut_dict"] = samplear_documentos_por_rut(
        all_data["_rut_dict"], ruts, args.tipo_muestreo , max_docs
    )

    if not all_data["_rut_dict"]:
        logging.warning("No se encontraron datos después del muestreo.")
        return {}, {}

    # Diccionarios de salida
    respuestas_por_rut = {}
    prompts_por_rut = {}

    semaphore = asyncio.Semaphore(args.inner_workers)  # Limita el número de llamadas concurrentes

    # --- Proceso interno para un RUT ---
    async def process_rut(rut: str) -> None:
        """Procesa todos los documentos de un RUT de forma secuencial."""
        textos_emisor = all_data["_rut_dict"].get(rut, {}).get("emisor", [])
        if not textos_emisor:
            return

        # Generar prompts de completación de texto
        prompts = [
            generar_prompt_completar_texto(
                extraer_info_concatenada(texto_legible_y_anonimo(txt, False))
            )
            for txt in textos_emisor]
        
        prompts_por_rut[rut] = prompts
        
        api_key= OPENAI_API_KEY if args.llm_model=="gpt-4o" else DEEP_API_KEY #que apikey usar segun modelo deepseek u openai
        url= URL_GPT if args.llm_model=="gpt-4o" else URL_DEEP    

        # Llamar a la API de forma concurrente
        async with aiohttp.ClientSession() as session:
            tasks = [
                _async_call_llm(session, p, model, temp, api_key,url, semaphore)
                for p in prompts
            ]
            responses = await asyncio.gather(*tasks)

        # Filtrar solo respuestas válidas
        responses = [resp for resp in responses if resp and not resp.startswith("Error:")]

        # Guardar resultados intermedios
        respuestas_por_rut[rut] = responses
        output = {
            "rut": rut,
            "giros_declarados_rut": all_data["rubros_por_rut"].get(rut),
            "documentos_emisor_original": all_data["_rut_dict"].get(rut, {}).get("emisor"),
            "documentos_receptor_original": all_data["_rut_dict"].get(rut, {}).get("receptor"),
            "completaciones_emisor_limpias": responses,
            "completaciones_receptor_limpias": [],
        }
        guardar_pickle(output, f"salida_rubro_{rut}.pkl", RESULTS_DIR)
        
        # Crear prompt para clasificación económica
        prompt_class = generar_prompt_clasificacion(
            output.get('completaciones_emisor_limpias', []),
            output.get('completaciones_receptor_limpias', []), # SOLO EMISOR por ahora
            RESUMEN_RUBROS_ADICIONALES,
            output.get('giros_declarados_rut', []),
            generar_prompt2
        )
        
        # Llamada sincrónica para clasificación
        response_json = call_llm(prompt_class, model , temp, api_key, url)
        logging.info(f"Clasificación recibida: {response_json}")

        try:
            response_json = extraer_contenido_entre_llaves(response_json)
        except Exception as e:
            logging.error(f"Error procesando clasificación para {rut}: {e}")
            response_json = None
        
        # Si hubo clasificación, guardar
        if response_json:
            output['clasificacion_economica'] = response_json.get("main_rubros", ["UNKNOWN_RUBRO"])
            output['justification'] = response_json.get("justification", "Respuesta no procesada")
        else:
            output['clasificacion_economica'] = ["API_ERROR"]
            output['justification'] = "Error en llamada a la API"

        guardar_pickle(output, f"clasificacion_{rut}.pkl", CLASSIFICATION_RESULTS_DIR)

    # Ejecutar en paralelo por RUT
    await async_tqdm.gather(*(process_rut(r) for r in ruts), desc="Procesando RUTs")

    logging.info("Proceso completado para todos los RUTs.")
    return respuestas_por_rut, prompts_por_rut



# --- MAIN ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fase 1: Extracción y completación de textos.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--rut", type=str, help="Un solo RUT para procesar.")
    input_group.add_argument(
        "--rut-list-path", nargs='+', type=str, help="Una o más rutas a archivos con listas de RUTs."
    )

    parser.add_argument("--new-bucket-data", action="store_true", help="Si se buscan datos directamente en S3.")
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL_NAME_API, help="Nombre del modelo LLM a usar.")
    parser.add_argument("--llm-temperature", type=float, default=LLM_TEMPERATURE)
    parser.add_argument("--max-docs-per-rut", type=int, default=5, help="Máximo número de documentos a procesar por RUT.")
    parser.add_argument("--solo-un-rubro", action="store_true", help="Limita a un solo rubro del SII por RUT.")
    parser.add_argument("--inner_workers", type=int, default=INNER_WORKERS, help="Workers para llamadas a la API dentro de un RUT.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CLASSIFICATION_RESULTS_DIR,
        help="Directorio donde se guardarán los resultados de la clasificación."
    )
    parser.add_argument("--tipo_muestreo", type=str, default="aleatorio", help="Tipo de muestreo sobre textos de cliente.")

    args = parser.parse_args()


    # --- Obtener lista de RUTs ---
    if args.rut:
        ruts = [args.rut.upper()]
    else:
        ruts = []
        for path in args.rut_list_path:
            ruts.extend(load_ruts_from_file(path))
        ruts = sorted(list(set(ruts)))  # Eliminar duplicados
        logging.info(f"Total RUTs únicos a procesar: {len(ruts)}")

    if not ruts:
        logging.warning("No hay RUTs para procesar. Finalizando.")
        sys.exit(0)

    # --- Función principal asíncrona ---
    async def run():
        respuestas, prompts_generados = await ejecutar_prueba_async(
            ruts=ruts,
            model=args.llm_model,
            temp=args.llm_temperature,
            max_docs=args.max_docs_per_rut,
            args=args,
            solo_un_rubro=args.solo_un_rubro,
            new_bucket_data=args.new_bucket_data
        )

        logging.info("Resultados de la ejecución:")
        for rut, prompts in prompts_generados.items():
            logging.info(f"RUT {rut}: {len(prompts)} prompts generados.")

    # Ejecutar
    asyncio.run(run())
