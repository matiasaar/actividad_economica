# llm/prompts.py

from config import RESUMEN_RUBROS_ADICIONALES # Para acceder a los resúmenes de rubros

def generar_prompt_completar_texto(texto_incompleto: str) -> str:
    """
    Genera el prompt para la tarea de completar/corregir texto.
    """
    prompt = f"""
Corrige o completa el siguiente texto en español. El texto contiene el nombre del comprador y vendedor, productos comprados, y otra información contextual como el giro del comprador y si la venta es a consumidor final.
Extrae los siguientes campos del contexto y preséntalos en formato `clave:valor`, uno por línea. No incluyas títulos, viñetas ni texto adicional. Las claves a usar son: vendedor, comprador, fecha, monto_total, producto, cantidad, monto_item.

1. Añade contexto relevante a los productos. Si es poco informativo, corrige o ajusta los productos para que sean más lógicos.
2. Asegúrate de que el **giro del comprador tenga coherencia con su nombre**. Si no es creíble, corrige o ajusta el giro para que sea más lógico.
3. Si la venta es a consumidor final, tenlo en cuenta para ajustar el lenguaje o detalles del giro.
4. El texto resultante debe estar completamente en español, limpio y con sentido.
5. Si el contenido del detalle del producto comprado viene sin información, retorna el comentario "sin información relevante". Ejemplo: si en detalle se detalla " " retorna "sin información relevante"

Texto original: "{texto_incompleto}"

Importante: responde únicamente con el texto corregido en español, sin explicaciones, títulos ni traducciones.

Texto corregido:
"""
    return prompt



def generar_prompt_clasificacion(texts_emisor: list, texts_receptor: list,
                                 resumen_rubros: dict, rubros_rut: list,
                                 generar_prompt2_func) -> str:
    """
    Genera un prompt para clasificar textos basado en interacciones emisor/receptor y rubros.
    Esta función es un marcador de posición y debe ser refinada basándose en la lógica de clasificación real.
    `generar_prompt2_func` es un sustituto temporal para  `generar_prompt2`.
    """
    emisor_info = "\n".join(texts_emisor) if texts_emisor else "No hay textos de emisor."
    receptor_info = "\n".join(texts_receptor) if texts_receptor else "No hay textos de receptor."
    current_rubros_str = ", ".join(rubros_rut) if rubros_rut else "No se conocen rubros actuales."

    # generar_prompt2_func (contiene reglas)
    additional_context_prompt = generar_prompt2_func(emisor_info, receptor_info)

    # Integrar RESUMEN_RUBROS_ADICIONALES de config si es relevante para la clasificación
    rubro_descriptions = "\n".join([f"- {k}: {v}" for k, v in resumen_rubros.items()])

    prompt = f"""
Se requiere clasificar la actividad económica de un RUT basándose en sus interacciones como emisor (ventas) y receptor (compras) de documentos.

Ventas del RUT (emisor):
{emisor_info}

Compras del RUT (receptor):
{receptor_info}

Rubros que el RUT declara tener: {current_rubros_str}

Resumen de Rubros Económicos de Referencia:
{rubro_descriptions}

{additional_context_prompt}

Basado en esta información, ¿cuáles son los rubros principales a los que este RUT pertenece? Proporciona una justificación concisa.


"""
    return prompt

def generar_prompt2(arg1, arg2):
    """
    Genera el segmento de prompt de contexto adicional para la clasificación
    basado en las reglas específicas para la asignación de rubros.
    arg1: textos del emisor (limpios)
    arg2: textos del receptor (limpios)
    """
    # En este contexto, arg1 y arg2 ya son las listas de textos limpios,
    # aunque el prompt original los llama "textos emisor" y "textos receptor"
    # en el prompt de clasificación. La función genera un segmento de CONTEXTO,
    # no un prompt completo.
    
    # NOTA: Dentro de este prompt, las reglas 6, 7 y 8 requieren
    # acceso a la información de los textos completos (nombre de vendedor/comprador,
    # detalle del producto). El prompt se está construyendo para que el LLM
    # tenga esas instrucciones. El LLM será quien "analice" y "relacione".

    context_prompt = '''
Con esta información y los detalles de los documentos:

1. **Analiza cada documento** individualmente en los segmentos de "Información del RUT (emisor o vendedor)" e "Información del RUT (receptor o comprador)".
2. **Identifica el rol del RUT bajo análisis en CADA DOCUMENTO**: si en un documento el RUT es el "Nombre del vendedor", analiza lo que vende; si en un documento el RUT es el "Nombre del comprador", analiza lo que compra.
3. **Relaciona el contenido del documento** (especialmente los "Productos vendidos", "Giro del comprador", y si es "Venta a consumidor final") con una o más actividades económicas del "Resumen de Rubros Económicos de Referencia" que se te proporciona.
4. **Compara** la(s) actividad(es) económica(s) que asignes con los "Rubros que el RUT declara tener" que se te proporcionan. 

---

### Reglas especiales (Prioridad Alta):

6. Si el "Nombre del vendedor" o el "Nombre del comprador" en **cualquier documento relevante** corresponde a una entidad pública, estatal, ministerial, municipal, o de regulación nacional (ej. "Municipalidad de...", "Ministerio de...", "Servicio de...", "Tesorería General de..."), asigna **prioritariamente** el siguiente rubro:

   **"ADMINISTRACION PUBLICA Y DEFENSA; PLANES DE SEGURIDAD SOCIAL DE AFILIACION OBLIGATORIA"**
   Y justifica específicamente con el nombre de la entidad pública encontrada.

7. Si el "Productos vendidos" en un documento específico contiene **exactamente** "sin información relevante", basa la clasificación de ese documento principalmente en el **"Nombre del vendedor" o "Nombre del comprador"** (según el rol del RUT en ese documento) y el "Giro del comprador" (si es relevante).

8. Si el "Nombre del vendedor" o "Nombre del comprador" en **cualquier documento relevante** es **claramente informativo** y sugiere un rubro específico (ej: "Universidad de Chile" -> "ENSEÑANZA", "Hospital del Trabajador" -> "ACTIVIDADES DE ATENCION DE LA SALUD HUMANA Y DE ASISTENCIA SOCIAL"), asigna el rubro directamente relacionado con ese nombre del resumen, **incluso si el resto del contenido del documento es ambiguo o "sin información relevante"**. Prioriza la información del nombre si es muy clara.

9. Responde en formato JSON: {{"main_rubros": ["Rubro1", "Rubro2"], "justification": "Razon de la clasificacion"}}
10.  El valor de `main_rubros` debe ser una lista de strings. Si no puedes determinar un rubro con certeza, devuelve una lista vacía `[]`.
11.  El valor de `justification` debe ser una cadena de texto explicando concisamente tu razonamiento.
12. Tu salida debe ser **ÚNICA Y EXCLUSIVAMENTE un objeto JSON válido**.
13. Los **rubros declarados por el rut pueden ser erroneos**, ya que el rut puede estar mintiendo o por desconocimiento.
14. Los productos que un rut puede vender, **no necesariamente son fabricados por el**. Para asignar rubro manufactura asegurate muy bien de los productos sean fabricados por el rut.

'''
    return context_prompt
    #return f"Contexto adicional generado para clasificación: {arg1[:50]}... y {arg2[:50]}..."