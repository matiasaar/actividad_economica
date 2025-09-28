#run_completion.py

import argparse
import asyncio
import aiohttp
import logging
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Dict, Any

# --- Configuración de logging global ---
logging.basicConfig(
    level=logging.INFO,  # Cambiar a DEBUG para más detalle
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Importaciones del proyecto ---
from config import (
    TEXT_DATA_FILENAME, ACTIVITY_CODES_FILENAME, SII_DATA_FILENAME,
    RESULTS_DIR, RESUMEN_RUBROS_ADICIONALES,
    LLM_MODEL_NAME, LLM_TEMPERATURE, INNER_WORKERS, OUTER_WORKERS,
    OLLAMA_BASE_URL
)
from data.loader import LoadTexts, load_activity_codes_data, load_sii_data_complete, load_data_and_preprocess
from data.preprocessor import (
    texto_legible_y_anonimo, extraer_info_concatenada,
    map_codes_to_rubros, extract_ruts_and_giros_from_texts_codes,
    build_rut_text_dictionary, obtener_rubros_por_rut 
)

from llm.prompts import generar_prompt_completar_texto
from utils.helpers import *

# --- FUNCIONES AUXILIARES ---

async def _async_call_aiohttp(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    temp: float,
    semaphore: asyncio.Semaphore
) -> str:
    """
    Realiza una llamada individual a la API de Ollama de forma asíncrona.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temp, "top_p": 1, "repeat_penalty": 1.1, "num_ctx": 4200}
    }
    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None)
    
    async with semaphore:
        try:
            async with session.post(url, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                return data["message"]["content"].strip()
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
            async_tqdm.write(f"-- Error en llamada aiohttp para prompt '{prompt[:30]}...': {e}")
            return "Error: Fallo en la llamada a la API"


async def _process_completions(
    session: aiohttp.ClientSession,
    prompts: List[str],
    model: str,
    temp: float,
    semaphore: asyncio.Semaphore
) -> List[str]:
    """
    Procesa en paralelo múltiples prompts contra la API del LLM.
    """
    tasks = [_async_call_aiohttp(session, p, model, temp, semaphore) for p in prompts]
    return await asyncio.gather(*tasks)


# --- PIPELINE PRINCIPAL DE COMPLETACIÓN ---

async def run_completion_step(
    ruts: List[str],
    common_data: Dict[str, Any],
    args: argparse.Namespace
) -> Dict[str, Dict[str, List[str]]]:
    """
    Ejecuta la fase de completación de textos para un lote de RUTs.
    """
    logging.info(f"--- Ejecutando fase de COMPLETACIÓN para {len(ruts)} RUTs...")
    outer_semaphore = asyncio.Semaphore(args.outer_workers)
    results: Dict[str, Dict[str, List[str]]] = {rut: {'emisor': [], 'receptor': []} for rut in ruts}

    async with aiohttp.ClientSession() as session:
        
        async def process_rut(rut: str) -> None:
            """
            Procesa de manera aislada un solo RUT. RECORDAR QUE POSEE MAS DE UN TEXTO ASOCIADO
            """
            try:
                async with outer_semaphore:
                    texts_emisor_all = common_data['_rut_dict'].get(rut, {}).get('emisor', [])
                    limit = args.max_docs_per_rut
                    texts_emisor = texts_emisor_all[:limit] if limit is not None else texts_emisor_all
                    
                    if not texts_emisor:
                        return

                    if limit is not None and len(texts_emisor_all) > limit:
                        async_tqdm.write(f"--- RUT {rut}: Procesando {len(texts_emisor)}/{len(texts_emisor_all)} documentos.")

                    prompts = [
                        generar_prompt_completar_texto(extraer_info_concatenada(texto_legible_y_anonimo(txt, False)))
                        for txt in texts_emisor
                    ]
                    logging.debug(f"Prompts generados para RUT {rut}: {len(prompts)}")

                    if not prompts:
                        return

                    inner_semaphore = asyncio.Semaphore(args.inner_workers)
                    responses = await _process_completions(session, prompts, args.llm_model, args.llm_temperature_toContext, inner_semaphore)

                    results[rut]['emisor'] = OnlyAnswer([r for r in responses if not r.startswith("Error:")])
                    results[rut]['receptor'] = []

            except Exception as e:
                async_tqdm.write(f"--- ERROR CRÍTICO procesando RUT {rut}: {e}. Continuando con el siguiente.")
        
        await async_tqdm.gather(*[process_rut(rut) for rut in ruts], desc="Procesando Textos (Completación)")
        
    return results


# --- FUNCIÓN PRINCIPAL ---

async def main() -> None:
    """
    Orquesta todo el proceso.
    """
    parser = argparse.ArgumentParser(description="Fase 1: Extracción y completación de textos.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--rut", type=str, help="Un solo RUT para procesar.")
    input_group.add_argument("--rut-list-path", nargs='+', type=str, help="Una o más rutas a archivos con listas de RUTs.")

    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL_NAME)
    parser.add_argument("--llm-temperature-toContext", type=float, default=LLM_TEMPERATURE)
    parser.add_argument("--max-docs-per-rut", type=int, default=None)
    parser.add_argument("--solo-un-rubro", action="store_true")
    parser.add_argument("--inner_workers", type=int, default=INNER_WORKERS)
    parser.add_argument("--outer_workers", type=int, default=OUTER_WORKERS)
    parser.add_argument("--new-bucket-data", action="store_true")
    parser.add_argument("--tipo_muestreo", type=str, default="aleatorio")

    args = parser.parse_args()

    if args.rut:
        ruts = [args.rut.upper()]
    else:
        ruts = []
        for path in args.rut_list_path:
            ruts.extend(load_ruts_from_file(path))
        ruts = sorted(list(set(ruts)))
        logging.info(f"--- Total RUTs únicos a procesar: {len(ruts)}")

    if not ruts:
        logging.warning("--- No hay RUTs para procesar. Finalizando.")
        return

    common_data = load_data_and_preprocess(args, ruts)
    common_data["_rut_dict"] = samplear_documentos_por_rut(
        common_data["_rut_dict"], ruts, args.tipo_muestreo, args.max_docs_per_rut
    )

    total_batches = -(-len(ruts) // args.batch_size)
    for i in range(0, len(ruts), args.batch_size):
        batch = ruts[i:i + args.batch_size]
        logging.info(f"--- Procesando Lote {i//args.batch_size + 1}/{total_batches} ({len(batch)} RUTs) ---")
        
        completions = await run_completion_step(batch, common_data, args)

        logging.info(f"--- Guardando resultados del Lote {i//args.batch_size + 1} ---")
        for rut, data in completions.items():
            if data['emisor']:
                output = {
                    'rut': rut,
                    'giros_declarados_rut': common_data['rubros_por_rut'].get(rut),
                    'documentos_emisor_original': common_data['_rut_dict'].get(rut, {}).get('emisor'),
                    'documentos_receptor_original': common_data['_rut_dict'].get(rut, {}).get('receptor'),
                    'completaciones_emisor_limpias': data['emisor'],
                    'completaciones_receptor_limpias': data['receptor'],
                }
                guardar_pickle(output, f"salida_rubro_{rut}.pkl", RESULTS_DIR)

    logging.info("------- Proceso completado para todos los lotes. -------")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.error("################## Proceso interrumpido por el usuario.")
