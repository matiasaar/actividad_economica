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
      curl -fsSL https://ollama.com/install.sh | sh
      ollama serve &
      ollama pull deepseek-r1:32b #(si se corre localmente solo instalar la version 14b no la 32b)
      ```
      **NOTA:**
        - Es muy importante que cada vez que se modifique las variables de entorno de ollama, el servidor de apague y se vuelva a servir (ollama serve &).
        - En un entorno con GPU, es recomendable que si el modelo no queda completamente cargado en GPU se apague y se prenda nuevamente el servidor. Para verificar que el modelo quede completamente cargado los logs deben cumplir número de layers=offload, ej: layers.model=65 layers.offload=65
        - Si lo anterior no se cumple se debe modificar OLLAMA_NUM_PARALLEL. Ollama carga automaticamente el modelo según el parámetro anterior. Como base, OLLAMA_NUM_PARALLEL=20 necesita al rededor de 100 V-ram (total, puede ser la suma de varias GPUs menores).  
        - Para apagar el servido ollama basta correr.
      ```bash
     pkill ollama
     ```
   ###  Ejemplos de uso:

   - run_completion.py: realiza la completacion/contextualizacion de documentos.
        ```bash
        python3 run_completion.py \
        --rut-list-path /opt/LLM_MODEL/ruts_prueba.txt \
        --batch-size 200 \
        --llm-model deepseek-r1:32b \
        --llm-temperature-toContext 0.25 \
        --max-docs-per-rut 5 \
        --inner_workers 5 \
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
        python clasificador.py \
       --input-path results \
       --output-dir results_clas \
       --llm-model deepseek-r1:32b \
       --temperature 0.25 \
        --rut-list /opt/LLM_MODEL/ruts_prueba.txt \
       --batch-size 15 \
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
### 1. Generar Claves SSH en Windows con PuTTYgen. Se deben crear 2 claves, una irá para Github y otra para la nube (Nodeshift)
1. Abre PuTTYgen. Si no lo tienes, descárgalo e instálalo junto con PuTTY desde [putty.org](https://www.putty.org/).
2. Genera un nuevo par de claves.
3. En el menú **Type of key to generate**, selecciona **RSA**.
4. En **Number of bits in a generated key**, usa **4096** para una mayor seguridad.
5. Haz clic en **Generate** y mueve el ratón aleatoriamente sobre la ventana para crear la clave.
6. Haz clic en **Save private key** para guardar tu clave privada.
7. Asigna una frase de contraseña (**passphrase**) fuerte cuando se te pida.  
8. Guarda el archivo con una extensión `.ppk` en un lugar seguro (por ejemplo, `C:\Users\TuUsuario\.ssh\`).
9. Copia tu clave pública: en la ventana de PuTTYgen, el texto que aparece en el cuadro de arriba, que comienza con `ssh-rsa` y termina con tu dirección de correo electrónico, es tu clave pública.
10. Copia todo ese texto al portapapeles.

### 2. Agregar la Clave Pública a GitHub
1. Inicia sesión en GitHub.
2. Haz clic en tu foto de perfil (esquina superior derecha) y selecciona **Settings**.
3. En el menú de la izquierda, ve a **SSH and GPG keys**.
4. Haz clic en el botón **New SSH key** o **Add SSH key**.
5. Dale un título descriptivo a tu clave (por ejemplo, "clave modelo act eco").
6. En el campo **Key**, pega el texto de la clave pública que copiaste de PuTTYgen.
7. Haz clic en **Add SSH key**.

**NOTA:** En Github se obtendrá un token a partir de lo anterior. Esto es la clave y usuario respectivo que se pedira cuando se clone el repo en la VM de la nube.

### 3. Se debe crear una cuenta de Docker hub y luego subir la imagen docker.

### 4. En Nodeshift: 
   - Agregar la otra Clave Pública a Nodeshift (basta agregar en Add new SSH key).
   - Seleccionar Custome Image, y en Docker Image utilizar el nombre la imagen docker en Docker Hub (debe incluir la versión de la imagen ej: marriolamaxxa/my-llm-analyzer:main-v1.0)
   - En Docker Repository Authentification, server=docker.io. Lo demás corresponde a autenticación de cada usuario. 

### 5. Despliegue VM NodeShift:
  On start script de NodeShift: 
   ```bash
    #!/bin/bash
    export OLLAMA_NUM_PARALLEL=20  #PARAMETRO IMPORTANTE QUE CONTROLA CUANTAS LLAMADAS EN PARALELO ACEPTA OLLAMA. 
    export OLLAMA_FLASH_ATTENTION=true
    apt update && apt install zip -y 

    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &
    ollama pull deepseek-r1:32b #(si se corre localmente solo instalar la version 14b no la 32b)
      
    echo "Iniciando servidor Ollama..."
    nohup ollama serve --n-gpu-layers -1 > /root/ollama.log 2>&1 &
    sleep 30

    echo "Clonando repositorio de código..."
    git clone https://github.com/matiasaar/actividad_economica.git /opt/LLM_MODEL
    cd /opt/LLM_MODEL

    echo "Descargando modelos LLM..."
    ollama pull deepseek-r1:32b
   ```

**NOTA:** Los pasos anteriores se pueden ingresar en On start script de Nodeshift o directamente en el PowerShell o CMD de la VM creada en NodeShift.

  Transferir datos a la VM usando SCP (se ejecuta de manera local):
   ```bash
    scp -P <puerto_ssh> -i <ruta_a_tu_clave_privada_ssh> <ruta_a_archivo_local> root@<ip_de_tu_vm>:/opt/LLM_MODEL/data_files/
   ```
    
  Comprimir archivos para enviar desde la VM al entorno local (se ejecuta en la nube):

  ```bash
    zip -r <ruta_donde_se_crea_el_zip> <ruta_al_archivo_a_transferir> #ej:zip -r /opt/LLM_MODEL/results.zip /opt/LLM_MODEL/results
   ```

 Transferir desde la VM al entorno local (se ejecuta de manera local):
 ```bash
 scp -P  <puerto_ssh> -i <ruta_a_tu_clave_privada_ssh> -r  root@<ip_de_tu_vm>:<ruta_al_archivo_a_transferir> <ruta_entorno_local_donde_copia_archivo> 
 ```
