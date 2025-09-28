#clasificador.py

import argparse
import asyncio
import aiohttp
import os
import logging
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Dict, Any, Optional

# --- Importaciones del proyecto ---
from llm.prompts import (generar_prompt_clasificacion, generar_prompt2)
from utils.helpers import (
    extraer_contenido_entre_llaves, guardar_pickle, load_ruts_from_file, cargar_datos,OnlyAnswer
)
from config import (
    OLLAMA_BASE_URL, RESULTS_DIR, CLASSIFICATION_RESULTS_DIR, LLM_MODEL_NAME, LLM_TEMPERATURE,
    OUTER_WORKERS, RESUMEN_RUBROS_ADICIONALES
)

# --- Configuración de logging ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# --- LÓGICA DE PROCESAMIENTO ASÍNCRONO ---
# =========================================================

async def _async_call_aiohttp(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    temp: float,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Llamada genérica a la API de Ollama usando aiohttp.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": temp, "num_ctx": 5000}
    }
    #timeout = aiohttp.ClientTimeout(total=300)
    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None)
    
    async with semaphore:
        try:
            async with session.post(url, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
              #  print('DATA',data)
                content = data.get("message", {}).get("content", "").strip()
                return extraer_contenido_entre_llaves(content) or {
                    "error": "Contenido JSON no encontrado",
                    "justification": "Error de parseo."
                }
        except Exception as e:
            async_tqdm.write(f"Error en la llamada aiohttp: {e}")
            return {"error": str(e), "justification": f"Error en la llamada a la API: {e}"}


async def run_classification_batch(
    rut_data_list: List[Dict[str, Any]],
    model: str,
    temperature: float,
    output_dir: str,
    workers: int
) -> None:
    """
    Clasifica un batch de RUTs en paralelo (1 prompt por RUT).
    """
    os.makedirs(output_dir, exist_ok=True)
    outer_semaphore = asyncio.Semaphore(workers)

    async with aiohttp.ClientSession() as session:

        async def classify_rut(rut_data: Dict[str, Any]) -> None:
            """
            Genera prompt de clasificación y guarda resultado en un pickle.
            """
            rut: str = rut_data.get('rut', 'RUT_DESCONOCIDO')
            prompt: str = generar_prompt_clasificacion(
                rut_data.get('completaciones_emisor_limpias', []),
                rut_data.get('completaciones_receptor_limpias', []),
                RESUMEN_RUBROS_ADICIONALES,
                rut_data.get('giros_declarados_rut', []),
                generar_prompt2
            )
            
            response_json: Dict[str, Any] = await _async_call_aiohttp(
                session, prompt, model, temperature, outer_semaphore
            )

            if response_json:
                rut_data['clasificacion_economica'] = response_json.get("main_rubros", ["UNKNOWN_RUBRO"])
                rut_data['justification'] = response_json.get("justification", "Respuesta no procesada")
            else:
                rut_data['clasificacion_economica'] = ["API_ERROR"]
                rut_data['justification'] = "Error en llamada a la API"

            guardar_pickle(rut_data, f"clasificacion_{rut}.pkl", output_dir)

        tasks = [classify_rut(data) for data in rut_data_list]
        for future in async_tqdm.as_completed(tasks, total=len(tasks), desc="Clasificando RUTs"):
            await future

    logging.info(f"Lote completado. {len(rut_data_list)} archivos guardados/actualizados en '{output_dir}'.")


# =========================================================
# --- PUNTO DE ENTRADA PRINCIPAL ---
# =========================================================

async def main() -> None:
    """
    Función principal: parsea argumentos, carga datos y ejecuta el pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Clasifica la actividad económica de RUTs a partir de archivos .pkl pre-procesados."
    )
    parser.add_argument("--input-path", type=str, required=True, help="Archivo ZIP o folder con los .pkl de entrada.")
    parser.add_argument("--rut-list", nargs='+', type=str, help="Rutas a archivos con listas de RUTs.")
    parser.add_argument("--output-dir", type=str, default=CLASSIFICATION_RESULTS_DIR, help="Directorio de salida.")
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=LLM_TEMPERATURE)
    parser.add_argument("--workers", type=int, default=OUTER_WORKERS)
    parser.add_argument("--batch-size", type=int, default=500)
    
    args = parser.parse_args()
    
    ruts: List[str] = []
    if args.rut_list:
        for path in args.rut_list:
            ruts.extend(load_ruts_from_file(path))
        ruts = sorted(list(set(ruts)))
    logging.info(f"Total RUTs únicos a procesar: {len(ruts)}")
    
    
    #datos_a_procesar: List[Dict[str, Any]] = cargar_datos_desde_zip(args.input_zip, ruts)
    datos_a_procesar: List[Dict[str, Any]] = cargar_datos(args.input_path, ruts)

    if not datos_a_procesar:
        logging.warning("No se encontraron datos para procesar. Finalizando.")
        return

    total_batches: int = -(-len(datos_a_procesar) // args.batch_size)
    for i in range(0, len(datos_a_procesar), args.batch_size):
        batch_data: List[Dict[str, Any]] = datos_a_procesar[i:i + args.batch_size]
        logging.info(f"Procesando Lote {i//args.batch_size + 1}/{total_batches} ({len(batch_data)} RUTs)")
        
        await run_classification_batch(
            rut_data_list=batch_data,
            model=args.llm_model,
            temperature=args.temperature,
            output_dir=args.output_dir,
            workers=args.workers
        )

    logging.info("Proceso completado para todos los lotes.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Proceso interrumpido por el usuario.")
