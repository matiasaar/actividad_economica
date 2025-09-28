# Guía de Despliegue y Ejecución: Proyecto Actividad Económica

Este documento describe el proceso para configurar, desplegar y ejecutar la pipeline de clasificación de RUTs en alguna actividad económica, utilizando un modelo de lenguaje grande (LLM) a través de Ollama o por medio de la API de OpenAI. Se asume que el despliegue se realizará en un entorno de máquina virtual con GPU (ej. NodeShift) utilizando una imagen Docker personalizada.
---
## 1. Estructura del Proyecto

```plaintext
data/                     # Módulos para cargar, limpiar y preprocesar datos
 ├─ extractor.py              # Funciones para extraer información de textos
 ├─ get_data_bucket.py        # Funciones para acceder a S3 y buckets
 ├─ loader.py                 # Funciones de carga de datos locales
 ├─ preprocessor.py           # Preprocesamiento de datos
 ├─ rubros.py                 # Funciones de manejo de rubros
 └─ sii_parser.py             # Parseo de datos del SII

data_files/               # Archivos de datos originales, grandes o sensibles (no subir a Git)
 ├─ actividades_rubro_subrubro_limpio.xlsx   # Mapeo códigos de actividad a rubros (facilitada por DP)
 ├─ textos_etiquetas_NEW_code.txt            # Textos con etiquetas para clasificación
 └─ v_sii_2.gzip                             # Datos completos del SII

llm/                      # Código para prompts y herramientas auxiliares
 └─ prompts.py                # Prompts definidos para LLM

utils/                    # Funciones auxiliares de uso general
 └─ helpers.py                # Funciones de utilidad (pickle, zip, JSON, etc.)

config.py                 # Variables globales de configuración
api_model.py              # Modelo completo usando API de OpenAI (flujo completo: contexto + asignación de rubro)
clasificador.py           # Modelo OLLAMA, realiza la asignación de un rubro
run_completion.py         # Modelo OLLAMA, realiza la contextualización de textos
Dockerfile                # Permite construir una imagen Docker con el entorno completo
requirements.txt          # Dependencias de Python necesarias
.env                      # Variables de entorno necesarias para la ejecución. Se deben configurar segun usuario. NO SUBIR A GIT. 
```
---
# 2. Como ejecutar los códigos (localmente y en la nube)
  ## 2.1 Modelo ollama
   - Definir variables de entorno
      ```bash
      export OLLAMA_NUM_PARALLEL=20    # numero de llamadas en paralelo a permitir por ollama. MUY IMPORTANTE CUANDO SE USA GPU. 
      export OLLAMA_FLASH_ATTENTION=true  #permite flash atenttion. Necesario para más rapidez de ejecición
      apt update && apt install zip -y #esto instala zip. Solo es relevante cuando se ejcuta el codigo en la nube. 
      ```    
   - Instalar ollama y modelos
      ```bash
      curl -fsSL https://ollama.com/install.sh | sh-
      ollama serve &
      ollama pull deepseek-r1:32b #(si se corre localmente solo instalar la version 14b no la 32b)
      ```
   ###  Ejemplos de uso:

   - run_completion.py: realiza la completacion/contextualizacion de documentos.
        ```bash
        python3 run_completion.py `
        --rut-list-path "C:\Users\mariola_maxxa\Desktop\Modelo_Actividad_Economica\ruts_prueba.txt" `
        --batch-size 200 `
        --llm-model deepseek-r1:14b `
        --llm-temperature-toContext 0.25 `
        --max-docs-per-rut 5 `
        --inner_workers 5 `
        --outer_workers 15
        ```
       - Argumentos de run_completion.py:
         ```bash
         --rut-list-path  #path al txt con ruts a preocesar. es un rut por linea
         --batch-size 200 #batch de ruts a procesar en una iteracion
         --llm-model deepseek-r1:14b #modelo llm a usar. si se tiene gpu correr deepseek-r1:32b
         --llm-temperature-toContext  0.25 #temperatura del llm. no elegir algo superior a 0.2
         --max-docs-per-rut 5 #numero maximo de documentos de ventas a considerar por rut
         --inner_workers 5  #numero de llamadas en paralelo intra-rut  (corre en paralelo los textos asociados a un rut)
         --outer_workers  15  #numero de llamadas en paralelo inter-rut (corre en paralelo multiples rut)
         --solo-un-rubro #arg. de tipo store true. si NO se agrega, se consideran todos los rubros de un rut en el SII y boletas, sino, se considera 1 solo
         --new-bucket-data # arg. de tipo store true. si no se agrega, los documentos de un rut se obtiene del documento textos_etiquetas_NEW_code.txt. Si se agrega, el codigo descarga los datos directamente desde la carpeta asociada al rut en el bucket. 
         --tipo_muestreo # tipo de muestreo a realizar sobre los documenots. por defecto es "aleatorio".
         ```
        
   - clasificacion.py: realiza la asignacion de un rubro. El rut debe haber pasado por el paso previo (run_comlpetion.py)
        ```bash
        python clasificador.py `
       --input-path results `
       --output-dir results_clas `
       --llm-model deepseek-r1:14b `
       --temperature 0.25 `
        --rut-list "C:\Users\mariola_maxxa\Desktop\Modelo_Actividad_Economica\ruts_prueba.txt" `
       --batch-size 15 `
       --workers 20
        ```
        
       - Argumentos de clasificacion.py:
        ```bash
       --input-path # . zip que contiene .pkl con el json obtenido en  run_completion.py
       --output-dir results_clas ` #nombre de la carpeta donde se guardaran los resultados finales
       --llm-model deepseek-r1:14b `  #nombre del modelo. Local 14b y nube 32b 
       --temperature 0.25 `  #temperatura del llm. no elegir algo superior a 0.2
        --rut-list "C:\Users\mariola_maxxa\Desktop\Modelo_Actividad_Economica\ruts_prueba.txt" `
       --batch-size 15 `  #batch de ruts a procesar en una iteracion
       --workers 20 #numero de llamadas en paralelo a ollama
        ```
        
  ## 2.2 Modelo api
   -  Instalar API de OpenAI
       ```bash
       pip install -q openai
       ```

   ###  Ejemplo de uso:
   ```bash
     python api_model.py
     --rut-list-path  #path al txt con ruts a preocesar. es un rut por linea
   ```
     
   - otros argumentos posibles de api_model.py:
     ```bash
         - new-bucket-data #Activa un modo donde los datos se buscan directamente en S3.
         - llm-model #Nombre del modelo LLM que se usará para el procesamiento. ("gpt-4o" o 'deepseek-reasoner')
         - llm-temperature #Valor de temperatura para el modelo LLM (controla creatividad/aleatoriedad en las respuestas).
         - max-docs-per-rut #Número máximo de documentos a procesar por cada RUT (límite por cliente).
         - solo-un-rubro #Restringe el procesamiento a un solo rubro del SII por RUT.
         - inner_workers #Número de workers (procesos/hilos) usados para llamadas a la API dentro de un mismo RUT.
         - output-dir #Directorio donde se guardarán los resultados de la clasificación.
         - tipo_muestreo #Define el tipo de muestreo aplicado sobre los textos del cliente (por defecto, “aleatorio”).       
     ```     
--- 
# 3. Pasos Para Correr en la nube (NodeShift)

- Preparacion_entorno_local:
  - Python 3.10+ y pip.
  - Docker Desktop para construir y gestionar imágenes Docker.
  - Git y Git Bash para gestionar el código y usar comandos SSH.
  - PuTTYgen para generar claves SSH (ppk y OpenSSH). **Crear dos keys SSH**. Una irá asociada a Github y la otra a la nube (Nodeshift).
  - Cuentas activas en GitHub y Docker Hub. Luego, se debe subir a Docker Hub la imagen docker creada del proyecto. 

### Despliegue VM NodeShift:
  On start script de NodeShift: 

    #!/bin/bash
    export OLLAMA_NUM_PARALLEL=20  #PARAMETRO IMPORTANTE QUE CONTROLA CUANTAS LLAMADAS EN PARALELO ACEPTA OLLAMA. 
    export OLLAMA_FLASH_ATTENTION=true
    apt update && apt install zip -y 
 
    echo "Iniciando servidor Ollama..."
    nohup ollama serve --n-gpu-layers -1 > /root/ollama.log 2>&1 &
    sleep 30

    echo "Clonando repositorio de código..."
    git clone https://github.com/matiasaar/actividad_economica.git /opt/LLM_MODEL
    cd /opt/LLM_MODEL

    echo "Descargando modelos LLM..."
    ollama pull deepseek-r1:32b

    echo "Instalando dependencias Python..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    echo "Script de inicio completado. Transferir archivos de datos grandes vía SCP/SFTP."
    sleep infinity

NOTA: Los pasos anteriores se pueden ingresar en On start script de Nodeshift o directamente en el PowerShell o CMD de la VM creada en NodeShift.

  Transferir datos a la VM usando SCP:
  
    ```bash
    scp -P <puerto_ssh> -i <ruta_a_tu_clave_privada> <ruta_a_archivo_local> root@<ip_de_tu_vm>:/opt/LLM_MODEL/data_files/
    ```
    
  Transferir datos desde la VM al entorno local usano SCP:
