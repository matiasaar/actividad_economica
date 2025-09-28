# config.py

import os
from collections import defaultdict # Se añade porque se usa para estructuras por defecto

# --- Configuración General ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio raíz del proyecto

# --- Rutas de Datos ---
DATA_DIR = os.path.join(BASE_DIR, 'data_files') # Archivos de datos de entrada
RESULTS_DIR = os.path.join(BASE_DIR, 'results') # Archivos de resultados de salida. Esto para resultados de contexto.
 

CLASSIFICATION_RESULTS_DIR = os.path.join(BASE_DIR,'results_clas')# Archivos de resultados de salida de clasificacion 

# Asegurarse de que los directorios existan
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Nombres de Archivos ---
TEXT_DATA_FILENAME = "textos_etiquetas_NEW_code.txt" #contiene todos los textos sampleados desde el bucket. 
ACTIVITY_CODES_FILENAME = "actividades_rubro_subrubro_limpio.xlsx" #este y el siguiente son archivos para obtener rubros economicos asoc. a ruts.
SII_DATA_FILENAME = "v_sii_2.gzip"

# --- Configuración del LLM ---
LLM_MODEL_NAME = 'deepseek-r1:32b' #nombre del modelo en ollama
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 1
LLM_REPEAT_PENALTY = 1.1

LLM_MODEL_NAME_API = 'deepseek-reasoner' #nombre del modelo en api de deepsek
#URL_DEEP = "https://api.deepseek.com/v1/chat/completions"
#URL_GPT = "https://api.openai.com/v1/chat/completions"
URL_DEEP = "https://api.deepseek.com/v1"
URL_GPT = "https://api.openai.com/v1"

#--- Numero de Workers para procesamiento paralelo----
INNER_WORKERS=4 
OUTER_WORKERS=2




#-- URL de modelos---
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

#OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
#OLLAMA_BASE_URL = f"http://localhost:{OLLAMA_PORT}"


# --- Reemplazos para Procesamiento de Texto ---
# 
REEMPLAZOS_LEGIBLES = {
    "tipodte": "tipo de documento",
    "fchemis": "fecha de emision",
    "tpotrancompra": "tipo de transaccion de compra",
    "tpotranventa": "tipo de transaccion de venta",
    "fmapago": "forma de pago",
    "rznsocemisor": "razon social del emisor",
    "rznsocrecep": "razon social del receptor",
    "girorecep": "giro del receptor",
    "mntneto": "monto neto",
    "tasaiva": "tasa iva",
    "iva": "iva",
    "mnttotal": "monto total",
    "nrolindet": "linea del detalle",
    "nmbitem": "nombre del producto",
    "qtyitem": "cantidad del producto",
    "prcitem": "precio unitario",
    "montoitem": "monto del producto"
}

# --- Resúmenes de Rubros (para prompts o contexto de clasificación) ---
# Este es un diccionario grande; considerar si es que debería cargarse desde un archivo para gran escala
#
RESUMEN_RUBROS_ADICIONALES = {
    "ACTIVIDADES DE ATENCION DE LA SALUD HUMANA Y DE ASISTENCIA SOCIAL":
        "Servicios médicos, hospitalarios, dentales, laboratorios clínicos, asistencia a personas mayores, con discapacidad o en rehabilitación, centros de salud y asistencia social sin alojamiento.",
    "INFORMACION Y COMUNICACIONES":
        "Servicios de tecnologías de la información, desarrollo de software, telecomunicaciones, medios de comunicación (radio, TV, cine), edición de libros y periódicos, y portales web.",
    "COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACION DE VEHICULOS AUTOMOTORES Y MOTOCICLETAS":
        "Venta de productos al por menor y mayor (alimentos, ropa, tecnología), comercio por internet, venta y reparación de vehículos, combustibles y repuestos.",
    "INDUSTRIA MANUFACTURERA":
        "Transformación de materias primas en productos elaborados como alimentos, textiles, productos químicos, maquinaria, muebles, productos metálicos, vehículos y equipos electrónicos.",
    "ENSEÑANZA":
        "Servicios educativos en todos los niveles: preescolar, básica, media, superior, formación técnica y profesional, e instrucción especializada.",
    "AGRICULTURA, GANADERIA, SILVICULTURA Y PESCA":
        "Incluye actividades agrícolas como el cultivo de cereales, hortalizas, frutas, legumbres, semillas y plantas especiales; "
        "ganadería con la cría de aves, bovinos, ovinos, cerdos y otros animales; actividades forestales como extracción de madera,"
        " reforestación y servicios silvícolas; y pesca y acuicultura tanto marítima como de agua dulce, junto con servicios asociados. Eliminación de plagas agricolas",
    "TRANSPORTE Y ALMACENAMIENTO":
        "Comprende el transporte de carga y pasajeros por carretera, ferrocarril, aéreo, marítimo y vías interiores; servicios de apoyo al transporte (como terminales, agencias, carga y descarga); actividades de almacenamiento incluyendo frigoríficos y bodegas; y servicios postales y de mensajería.",
    "ACTIVIDADES PROFESIONALES, CIENTIFICAS Y TECNICAS":
        "Incluye servicios de consultoría, ingeniería, arquitectura, contabilidad, auditoría, asesoría legal, publicidad, diseño, estudios de mercado, investigación científica y técnica, veterinaria y actividades especializadas como traducción, fotografía y revisión técnica.",
    "ACTIVIDADES DE SERVICIOS ADMINISTRATIVOS Y DE APOYO":
        "Engloba servicios de alquiler de maquinaria, vehículos y equipos; limpieza, jardinería, fumigación de tipo no agricola; agencias de viajes y empleo; call centers; seguridad privada; apoyo administrativo y logístico a empresas; organización de eventos; actividades de cobranza y clasificación crediticia.",
    "CONSTRUCCION":
        "Cubre la construcción de edificios residenciales y no residenciales, obras de infraestructura como caminos, ferrocarriles y servicios públicos; preparación del terreno, demolición, instalaciones eléctricas y sanitarias, terminaciones, y otras obras especializadas de construcción.",
    "OTRAS ACTIVIDADES DE SERVICIOS":
        "Incluye servicios personales como peluquería, masajes, lavandería, servicios funerarios, reparación de bienes personales y del hogar, y actividades de asociaciones, sindicatos, organizaciones religiosas, políticas, culturales y sociales.",
    "ACTIVIDADES DE ALOJAMIENTO Y DE SERVICIO DE COMIDAS":
        "Comprende la operación de hoteles, moteles, residenciales, campings y otros alojamientos turísticos; además de restaurantes, servicios de banquetería, concesiones de alimentación, y bares o discotecas con servicio de bebidas.",
    "ACTIVIDADES INMOBILIARIAS":
        "Abarca la compra, venta y arriendo de bienes raíces, tanto amoblados como no amoblados; servicios de gestión inmobiliaria prestados por terceros; y servicios imputados de alquiler de viviendas.",
    "SUMINISTRO DE AGUA; EVACUACION DE AGUAS RESIDUALES, GESTION DE DESECHOS Y DESCONTAMINACION":
        "Incluye la captación y distribución de agua potable; la recolección, tratamiento y eliminación de desechos sólidos y líquidos (peligrosos y no peligrosos); reciclaje de materiales como papel, vidrio y metales; y actividades de descontaminacion ambiental.",
    "ACTIVIDADES FINANCIERAS Y DE SEGUROS":
        "Incluye bancos, financieras, aseguradoras, AFP, isapres, fondos de inversión, leasing, casas de cambio, bolsas, agentes de valores, clasificadoras de riesgo, administradores de tarjetas de crédito y servicios auxiliares como asesorías e intermediación financiera.",
    "EXPLOTACION DE MINAS Y CANTERAS":
        "Incluye la extracción y procesamiento de minerales metálicos y no metálicos como cobre, oro, plata, hierro, sal, carbón, piedra y arcilla, así como actividades de apoyo para la minería y explotación de petróleo y gas natural. Base estabilizado, gravilla o arena y relacionados.",
    "ACTIVIDADES ARTISTICAS, DE ENTRETENIMIENTO Y RECREATIVAS":
        "Comprende actividades culturales y recreativas como teatros, conciertos, museos, bibliotecas, clubes deportivos, artistas independientes, juegos de azar, casinos, parques temáticos, zoológicos, jardines botánicos y periodismo independiente.",
    "ADMINISTRACION PUBLICA Y DEFENSA; PLANES DE SEGURIDAD SOCIAL DE AFILIACION OBLIGATORIA":
        "Abarca la gestión pública, la defensa nacional, el orden público, regulación de servicios sociales y culturales, y los sistemas obligatorios de seguridad social. La emisión de documentos, certificados, derechos, servicios y permisos estatales. Otorgamiento de derechos.",
    "ACTIVIDADES DE ORGANIZACIONES Y ORGANOS EXTRATERRITORIALES":
        "Incluye las actividades de organismos internacionales y misiones diplomáticas extranjeras en el país.",
    "SUMINISTRO DE ELECTRICIDAD, GAS, VAPOR Y AIRE ACONDICIONADO":
        "Comprende la generación, transmisión y distribución de electricidad, la producción y suministro de gas, vapor, aire acondicionado e incluso la elaboración de hielo.",
    "ACTIVIDADES DE LOS HOGARES COMO EMPLEADORES; ACTIVIDADES NO DIFERENCIADAS DE LOS HOGARES":
        "Se refiere exclusivamente al empleo de personal doméstico por parte de los hogares."
}

# --- Marcadores de posición para componentes faltantes (definir o eliminar según sea necesario) ---
# Estos deberían ser inicializados/cargados dentro de las funciones específicas que los usan,
# o pasados como argumentos. Sin embargo, para un marcador de posición rápido, pueden estar aquí.
# 
RUTS_MALOS = ["12345678-9", "98765432-1"] # Ejemplo: Reemplazar con RUTs malos reales si son estáticos
# Esto probablemente debería cargarse desde un archivo o base de datos, no codificarse.
RUT_DICT_EXAMPLE = {
    "12345678-9": {
        "emisor": ["text_emisor_1", "text_emisor_2"],
        "receptor": ["text_receptor_A"]
    },
    "98765432-1": {
        "emisor": ["text_emisor_X"],
        "receptor": ["text_receptor_Y", "text_receptor_Z"]
    }
}
# Marcador de posición para mapear RUTs a sus rubros conocidos. 
RUBROS_POR_RUT_EXAMPLE = {
    "12345678-9": ["Industry1"],
    "98765432-1": ["Industry2", "Industry3"]
}