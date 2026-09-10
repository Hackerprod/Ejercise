### Yo

Es decir, tenemos lo que funciona y como funciona (Transformers), tenemos lo que no funciono y por que en la hera pre Transformer (RNN, CNN, etc), y sabemos que es lo mejor que puede hacer una cpu/ram apoyados de LLM que tienen mucho conocimiento

### GPT 5.6 Sol

Sí. Y eso cambia completamente la naturaleza del problema:

> **Ya no tendríamos que inventar una arquitectura desde cero. Podemos hacer ingeniería inversa de lo que funciona, usando como restricciones explícitas aquello que una CPU y la RAM hacen bien.**

Tenemos cuatro piezas que antes no existían juntas.

## 1. Un ejemplo positivo: el Transformer

El Transformer funciona y sabemos exactamente qué operaciones realiza. Podemos inspeccionar:

* representaciones internas;
* atención;
* activaciones del MLP;
* rutas residuales;
* cambios de estado entre capas;
* información conservada o descartada;
* comportamiento frente a contextos largos;
* diferencias entre tareas fáciles y difíciles.

El Transformer original mostró que eliminar la recurrencia permitía entrenar de manera mucho más paralela y conectar directamente posiciones lejanas mediante atención. No significa que la atención sea obligatoria para la inteligencia; significa que solucionó problemas fundamentales que las alternativas anteriores no resolvían simultáneamente. ([arXiv][1])

Por tanto, no tenemos que copiar su estructura. Debemos extraer sus **invariantes funcionales**:

```text
¿Qué capacidad proporciona la atención?
¿Qué capacidad proporciona el MLP?
¿Qué información conserva el residual?
¿Qué necesita realmente cada token?
¿Qué trabajo se repite innecesariamente?
```

La pregunta correcta deja de ser:

> “¿Cómo construimos otro Transformer?”

Y pasa a ser:

> “¿Cómo obtenemos las mismas capacidades con otra organización del cómputo?”

---

## 2. Un registro negativo: las limitaciones pre-Transformer

RNN, LSTM y CNN no fueron inútiles ni “fracasaron” completamente. Funcionaban. Pero tenían límites estructurales:

* estado oculto demasiado comprimido;
* dificultad para recuperar selectivamente información distante;
* dependencia secuencial durante entrenamiento;
* rutas largas entre elementos alejados;
* degradación al acumular información;
* operaciones relativamente homogéneas para todos los tokens.

Lo importante es que ahora conocemos **por qué** muchas alternativas eficientes no alcanzaban al Transformer.

Mamba, por ejemplo, identificó una debilidad central de varios modelos recurrentes, convolucionales y de espacio de estados: no bastaba con tener memoria lineal; era necesario que la propagación y el olvido fueran **dependientes del contenido de la entrada**. Al introducir selectividad, la arquitectura recuperó parte de las capacidades que antes favorecían a la atención. ([arXiv][2])

Hyena mostró algo parecido desde otra dirección: una convolución larga por sí sola no es suficiente, pero combinada con control dependiente de datos puede aproximarse mucho más a las capacidades de atención. ([arXiv][3])

Eso nos deja una lección importante:

> El problema de las arquitecturas antiguas no era simplemente que fueran recurrentes o convolucionales. El problema era que su memoria y su cómputo no eran suficientemente **selectivos, recuperables y dependientes del contenido**.

Por tanto, no debemos regresar a una RNN tradicional. Debemos rescatar:

* estado persistente de las RNN;
* localidad de las CNN;
* memoria explícita de los modelos asociativos;
* composición de sistemas simbólicos;
* recuperación por contenido de la atención;
* capacidad de entrenamiento moderno de los Transformers.

Pero reorganizado bajo otro contrato de hardware.

---

## 3. Un objetivo físico claro: CPU + RAM

La arquitectura debe diseñarse empezando por el hardware, no adaptar después el hardware al modelo.

El contrato podría ser:

```text
1. No leer todo el modelo por token.
2. Mantener un working set pequeño y caliente.
3. Ejecutar operaciones compactas vectorizables.
4. Hacer routing en bloques suficientemente grandes.
5. Reutilizar estado entre tokens.
6. Mantener el conocimiento total principalmente en RAM.
7. Expandir cómputo solo cuando la incertidumbre lo exija.
```

La métrica principal no debería ser solamente FLOPS.

Debería ser:

```text
calidad obtenida / bytes movidos desde RAM
```

Porque una CPU puede realizar muchos cálculos pequeños rápidamente si los datos están en caché. Lo que la destruye es traer repetidamente gigabytes de pesos para cada token.

Una arquitectura CPU-native debería separar cuatro magnitudes que hoy están demasiado unidas:

```text
Capacidad total del sistema
≠
Conocimiento residente en el working set
≠
Parámetros activados por token
≠
Bytes realmente leídos de RAM
```

Por ejemplo, de manera hipotética:

```text
Capacidad total:              equivalente a 40B
Working set de conversación:  300M
Ruta activa por token:        20M
Pesos nuevos leídos:          2–10M
```

Esos números concretos tendrían que demostrarse. Lo importante es la separación conceptual.

---

## 4. Ahora tenemos LLM como instrumento de investigación

Esta es probablemente la diferencia histórica más grande.

Un LLM puede ayudar a:

* revisar décadas de arquitecturas;
* identificar mecanismos repetidos;
* comparar papers;
* formular variantes;
* generar implementaciones;
* construir kernels;
* diseñar pruebas;
* analizar trazas;
* encontrar errores experimentales;
* proponer ablaciones;
* detectar que dos ideas aparentemente nuevas son equivalentes;
* convertir resultados en nuevas hipótesis.

Eso permite recorrer el espacio de arquitecturas mucho más rápido.

Pero hay una limitación seria:

> Un LLM conoce principalmente combinaciones de ideas que ya existen en sus datos de entrenamiento.

Puede producir nuevas combinaciones útiles, pero también:

* volver silenciosamente al Transformer;
* cambiar nombres sin cambiar mecanismos;
* proponer algo matemáticamente elegante pero lento;
* ignorar costes reales de caché;
* confundir plausibilidad con evidencia;
* afirmar novedad donde ya existe un paper similar.

Por eso el LLM debe actuar como **investigador asistente**, no como fuente de verdad. Cada propuesta debe terminar en código, medición y comparación.

---

# El Transformer puede convertirse en nuestro laboratorio

No necesitamos entrenar inmediatamente una arquitectura nueva desde cero. Podemos usar un Transformer pequeño como sistema de referencia.

La estrategia sería observarlo y preguntarnos:

```text
Para producir este token:

¿Qué capas cambiaron realmente la representación?
¿Qué neuronas o canales contribuyeron?
¿Qué atención fue necesaria?
¿Qué información estaba repetida?
¿Qué parte de la ruta podía predecirse desde el contexto anterior?
¿Qué cálculos podían sustituirse por estado persistente?
```

Entonces podemos intentar aprender una arquitectura que reproduzca **las transiciones útiles**, no necesariamente todos los cálculos del Transformer.

## Ejemplo experimental

Tenemos el estado de una capa:

$$
h_l
$$

Y el siguiente:

$$
h_{l+1}
$$

La transformación útil realizada por esa capa es:

$$
\Delta h_l = h_{l+1}-h_l
$$

En lugar de copiar todo el bloque Transformer, podemos investigar si esos cambios pueden representarse mediante un diccionario de pequeños operadores:

$$
\Delta h_l \approx \sum_{i \in S(x,s)} \alpha_i O_i(h_l,s)
$$

Donde:

* \(O_i\) son microoperadores;
* \(S(x,s)\) es un selector dependiente del token y del estado;
* solo se activan unos pocos operadores;
* \(s\) es un estado persistente de la conversación;
* los operadores activos pueden permanecer en caché.

El experimento real sería comprobar si miles de transformaciones densas aparentemente diferentes pueden descomponerse en una cantidad manejable de operaciones reutilizables.

Eso conecta directamente con tu idea de microexpertos, pero de una forma más verificable:

> En vez de decidir arbitrariamente que deben existir millones de microexpertos, primero observamos las transformaciones del modelo y medimos si realmente contienen estructura reutilizable y factorizable.

---

# Una arquitectura candidata

No necesariamente sería el diseño final, pero sí una hipótesis experimental razonable:

```text
Token de entrada
      │
      ▼
Codificador local pequeño
      │
      ▼
Estado persistente de conversación
      │
      ├────► Memoria relacional
      │
      ├────► Memoria episódica
      │
      └────► Working set contextual
      │
      ▼
Router jerárquico
      │
      ├── operador lingüístico
      ├── operador relacional
      ├── operador causal
      ├── operador numérico
      ├── operador de recuperación
      └── operador de planificación
      │
      ▼
Composición de pocos operadores
      │
      ▼
Verificador de confianza
      │
      ├── confianza alta → producir token
      └── confianza baja → ampliar working set
```

La clave sería que el router no seleccione entre expertos gigantes. Seleccionaría entre operaciones pequeñas, agrupadas jerárquicamente:

```text
dominio
  → capacidad
      → familia de operadores
          → operador concreto
```

Así se evita comparar directamente contra millones de opciones en cada token.

---

# Qué conservar del Transformer

No necesariamente sus bloques, sino sus capacidades:

### Recuperación dependiente del contenido

El modelo debe poder localizar una información porque es relevante para el token actual, no solamente porque ocurrió recientemente.

### Rutas cortas hacia información distante

No podemos obligar a que un dato atraviese miles de pasos recurrentes para volver a utilizarse.

### Representaciones distribuidas

No todo debe convertirse en símbolos rígidos. Los vectores siguen siendo esenciales para representar ambigüedad, similitud y generalización.

### Gradientes estables

Una arquitectura elegante que no pueda entrenarse a escala no sirve.

### Composición

Debe poder combinar capacidades conocidas para resolver situaciones no vistas.

### Escalabilidad

Añadir conocimiento debería aumentar capacidad sin obligar a ejecutar todo el conocimiento añadido.

Ese último punto sería la ruptura principal respecto al Transformer.

---

# Qué debemos rechazar

```text
Toda entrada atraviesa todas las capas.
Todo token lee casi todos los pesos activos.
Toda tarea usa aproximadamente la misma profundidad.
El contexto textual sirve como única memoria.
La capacidad total determina directamente el coste por token.
```

Una arquitectura CPU-native tendría que hacer exactamente lo contrario:

```text
Ruta condicional.
Profundidad dinámica.
Estado persistente.
Memoria recuperable.
Working set contextual.
Cómputo adaptativo.
Fallback cuando hay incertidumbre.
```

---

# El principal peligro técnico

La idea es coherente, pero existe un problema duro:

> Para seleccionar correctamente qué conocimiento activar, el sistema parece necesitar comprender primero la entrada; pero comprender la entrada puede requerir precisamente el conocimiento que todavía no ha activado.

Es el problema circular del router.

La posible solución no sería un único router perfecto, sino selección progresiva:

```text
1. Selección aproximada barata.
2. Ejecución de una ruta mínima.
3. Medición de incertidumbre.
4. Expansión de la ruta.
5. Verificación.
6. Fallback más completo cuando sea necesario.
```

También habría que impedir que los microoperadores produzcan millones de lecturas aleatorias. Una arquitectura extremadamente dispersa puede terminar siendo peor para CPU que una matriz densa pequeña.

Por eso la dispersión debe ser **estructurada**:

* operadores agrupados;
* páginas contiguas;
* selección a nivel de bloques;
* working sets persistentes;
* prefetch;
* reutilización durante varios tokens;
* routing jerárquico;
* caché guiada por tema y estado.

---

# El programa de investigación ya puede formularse

Tenemos:

```text
Transformer       = evidencia positiva y maestro
Pre-Transformer   = mapa de fallos conocidos
CPU/RAM           = restricciones físicas
LLM               = generador y analizador de hipótesis
Runtime           = entorno experimental
Benchmarks        = juez
```

Y el objetivo puede expresarse claramente:

> Construir una arquitectura capaz de conservar las propiedades funcionales esenciales del Transformer, mientras desacopla la capacidad total del modelo de los bytes leídos y del cómputo ejecutado por token.

Las métricas reales deberían ser:

```text
Pérdida/perplejidad
Calidad en razonamiento y recuperación
Bytes de DRAM leídos por token
Parámetros activos por token
Fallos de caché L3
Latencia batch 1
Tamaño del estado persistente
Coste al cambiar de tema
Frecuencia de fallback
Degradación con contexto largo
```

Ese marco elimina gran parte de la especulación. No buscamos una arquitectura “bonita”. Buscamos una arquitectura que gane simultáneamente en:

$$
\text{calidad},\quad
\text{bytes/token},\quad
\text{latencia},\quad
\text{capacidad escalable}
$$

Así que sí: **por primera vez tenemos al mismo tiempo el modelo que demuestra qué capacidades son posibles, el historial de lo que falló, un objetivo de hardware definido y modelos capaces de ayudarnos a explorar la solución**.

Eso no garantiza que encontremos la arquitectura correcta, pero significa que ya no estamos tanteando a oscuras. Podemos convertir el problema en una búsqueda experimental guiada y falsificable.

[1]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"
[2]: https://arxiv.org/abs/2312.00752?utm_source=chatgpt.com "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
[3]: https://arxiv.org/abs/2302.10866?utm_source=chatgpt.com "Hyena Hierarchy: Towards Larger Convolutional Language Models"

### Claude Opus 5

## La idea

El documento comete un error que se paga entero en CPU: propone dispersión **espacial** (muchos operadores diminutos, se activan pocos). Y la dispersión espacial fina en RAM es una trampa, porque el ahorro nominal de bytes se lo come el patrón de acceso. Leer 50 vectores dispersos de 8 KB no cuesta 400 KB de tiempo, cuesta 50 latencias de DRAM más el desperdicio de líneas de caché parcialmente usadas. Puedes reducir 10× los bytes contados y perder velocidad.

La alternativa:

> **La dispersión debe estar en el eje temporal, no en el espacial.** No operadores pequeños, sino operadores grandes y contiguos que sirvan a muchos tokens consecutivos.

Un bloque grande leído una vez y reutilizado durante 32 tokens amortiza sus bytes por 32, y se lee de forma secuencial, que es el único régimen donde una CPU alcanza su ancho de banda nominal. El mismo factor de ahorro, con el patrón de acceso invertido.

## Por qué esto es plausible y no un truco

Tokens consecutivos del mismo párrafo son semánticamente vecinos. Es muy improbable que el conjunto de conocimiento necesario cambie por completo entre "la capital de" y "Francia". Si eso es cierto, existe un working set conversacional real, y la unidad de diseño deja de ser el token y pasa a ser la **ventana**.

Eso convierte el modelo en algo con una analogía exacta: un sistema operativo con paginación por demanda. El router es el pager. Los operadores son páginas. El "working set contextual" del documento deja de ser una metáfora y pasa a ser el working set de Denning, con su literatura, sus políticas de residencia y su fallo de página medible.

## Lo que esto arregla del diseño del documento

**El router circular deja de ser circular.** El documento lo llama "el principal peligro técnico": para elegir qué activar hace falta entender, y para entender hace falta lo que aún no se ha activado. Falso dilema. El router no decide desde el token actual, predice desde el estado del token anterior qué bloques harán falta, hace prefetch, y verifica. Un fallo cuesta una lectura de DRAM, no una respuesta mala. Es un predictor de saltos, no un compilador.

**El router jerárquico de cuatro niveles sobra.** Dominio → capacidad → familia → operador son cuatro cargas dependientes, o sea pointer chasing, que es exactamente lo que destroza a una CPU. Un solo nivel plano, con tabla que quepa en L2, eligiendo entre unos cientos de bloques grandes.

**El experimento de Δh responde a la pregunta equivocada.** Descomponer Δh en un diccionario disperso mide si hay estructura reutilizable. No mide si se mueven menos bytes. Son cosas distintas y solo la segunda decide.

## Las dos medidas que deciden todo

Antes de diseñar nada:

**1. Curva de ancho de banda efectivo contra tamaño de bloque** en tu máquina: 64 B, 256 B, 1 KB, 4 KB, 64 KB, 1 MB, 16 MB, con acceso aleatorio entre bloques. Treinta líneas de C. Esa curva te da el **tamaño mínimo de operador** por debajo del cual la dispersión deja de pagar. La arquitectura se deriva de esa curva, no al revés.

**2. Coherencia temporal del conjunto activo** sobre cualquier transformer pequeño abierto. Por token y capa, las top-k neuronas del MLP que capturan el 90% de la norma de salida. Luego la unión sobre ventanas de W = 1, 8, 32, 128 tokens.

Umbrales fijados antes de mirar:

```
U(32)/N ≥ 0.60  → muerto. No hay working set, acabas leyendo todo igual.
U(32)/N ≤ 0.25  → el contrato es real y el layout de RAM se deriva de U(W).

```

La ganancia total es `(N·W / U(W)) × (BW_bloque / BW_secuencial)`. Los dos factores salen de esos dos experimentos, y ninguno requiere entrenar nada.

Si quieres, escribo cualquiera de los dos.

### GPT 5.6 Sol

Claude encontró una falla real del documento, pero **su sustitución tampoco puede aceptarse tal como está**. La formulación correcta no es:

> dispersión temporal en lugar de dispersión espacial

sino:

> **dispersión espacial gruesa, organizada en bloques contiguos, con persistencia temporal y reutilización demostrable en la jerarquía real de memoria.**

Su crítica mejora mucho la dirección, pero omite dos problemas físicos fundamentales.

# 1. Lo que Claude acertó

La objeción contra miles o millones de microoperadores dispersos es válida. En CPU, contar únicamente los bytes útiles puede ser engañoso. Importan también:

* tamaño de cada lectura;
* continuidad de las direcciones;
* cantidad de fallos de caché;
* paralelismo de memoria disponible;
* TLB;
* coste del router;
* formato de los pesos;
* trabajo realizado por cada bloque recuperado.

También acierta en que **la coherencia temporal debe medirse antes de diseñar la arquitectura**. Ya existe evidencia de que activaciones vecinas pueden persistir entre tokens. Un trabajo muy reciente, NeuroPrefetcher, reporta que aproximadamente el 82–85% de sus filas MLP activas persiste del token anterior al siguiente. LLM in a Flash usa una ventana de activaciones recientes y lecturas agrupadas; en sus pruebas, las lecturas de aproximadamente 32 KiB funcionaron mejor que transferencias neuronales minúsculas. ([arXiv][1])

También es correcto que el experimento de descomponer \(\Delta h\) no responde por sí solo a la pregunta importante. Puede demostrar que existe una base reutilizable, pero no demuestra que esa base:

* reduzca tráfico de DRAM;
* mejore latencia;
* sobreviva en caché;
* pueda seleccionarse económicamente;
* conserve la calidad.

Hasta ahí, Claude corrige correctamente el documento.

# 2. El error principal: residente en RAM no significa reutilizado por la CPU

Claude dice:

> Un bloque grande leído una vez y reutilizado durante 32 tokens amortiza sus bytes por 32.

Eso solo es cierto bajo una condición que no menciona:

> **El bloque debe permanecer en caché, o los 32 tokens deben procesarse mientras el bloque continúa caliente.**

En un modelo CPU normal, los pesos ya están en RAM. El cuello de botella es moverlos repetidamente:

$$
\text{DRAM} \rightarrow \text{LLC/L2/L1} \rightarrow \text{núcleo}
$$

Supongamos que una capa utiliza un bloque de 1 MB. Después de utilizarlo para el token \(t\), el modelo atraviesa el resto de las capas. Antes de volver a ese mismo bloque para \(t+1\), la CPU puede haber tocado decenas o cientos de megabytes adicionales.

Si la distancia de reutilización supera la capacidad útil de la caché de último nivel, ese bloque habrá sido expulsado. Para el token siguiente se vuelve a leer desde DRAM.

En ese caso:

```text
mismo bloque lógico durante 32 tokens
≠
una sola lectura física desde DRAM
```

Puede convertirse simplemente en:

```text
32 lecturas secuenciales del mismo bloque
```

Eso sigue siendo mejor que 32 lecturas aleatorias, pero **no produce una amortización de 32× en bytes**.

La reutilización temporal reduce realmente el tráfico cuando se cumple al menos una de estas condiciones:

1. El conjunto activo completo entre reutilizaciones cabe en LLC.
2. Se procesan varios tokens por cada carga de pesos.
3. Existe un nivel más lento que RAM, por ejemplo SSD/NVMe, y RAM actúa como búfer residente.
4. Existe memoria especializada o scratchpad controlable donde los bloques permanecen explícitamente.

Precisamente, LLM in a Flash y NeuroPrefetcher trabajan principalmente sobre el límite **flash/NVMe → DRAM o búfer GPU**, no sobre el límite puro **DRAM → caché CPU**. Mantienen las filas activas en una memoria residente y cargan solamente el delta desde almacenamiento. Esa evidencia respalda la idea temporal, pero no demuestra que una CPU reutilice los pesos desde LLC durante 32 pasos autoregresivos. ([arXiv][2])

## Consecuencia

El working set relevante no es solamente:

```text
qué bloques permanecen seleccionados
```

Debe ser:

```text
qué bloques permanecen físicamente en el nivel rápido
cuando vuelven a utilizarse
```

La analogía de Denning sería exacta para:

```text
SSD/NVMe = almacenamiento secundario
RAM      = memoria residente
```

Pero si el objetivo es acelerar un modelo que ya cabe en RAM, entonces la analogía debe bajar un nivel:

```text
RAM = almacenamiento lento
LLC = conjunto residente efectivo
```

Y allí el presupuesto puede ser de decenas de megabytes, no de cientos de megabytes o gigabytes.

# 3. La ventana solo amortiza pesos si podemos calcular una ventana

El decode autoregresivo exacto normalmente funciona así:

```text
token t
→ todas las capas
→ producir token t+1
→ todas las capas
→ producir token t+2
```

No podemos calcular arbitrariamente 32 tokens futuros con un operador ya cargado, porque todavía no conocemos esos tokens.

Para aprovechar de verdad un bloque sobre varios tokens antes de expulsarlo, necesitamos alguna de estas transformaciones:

### Prefill o procesamiento de varias secuencias

Los tokens ya son conocidos y pueden procesarse en lote. Aquí la ventana es natural.

### Speculative decoding o MTP

Un modelo pequeño propone varios tokens y el modelo grande los verifica juntos. Esto permite que una carga de pesos contribuya a varios tokens aceptados. Los sistemas de speculative decoding explotan precisamente esa amortización del modelo objetivo sobre varios tokens. ([arXiv][3])

### Arquitectura recurrente o de estado persistente nueva

El operador no tendría que atravesar todos los tokens como un Transformer ordinario. Podría actualizar un estado compacto y ser invocado solamente cuando cambia el régimen contextual.

Sin uno de esos mecanismos, “la unidad de diseño es la ventana” puede ser cierto para el **routing**, pero no necesariamente para el **cómputo ni el tráfico DRAM**.

# 4. La analogía con un page fault no es exacta

Claude afirma:

> Un fallo cuesta una lectura de DRAM, no una respuesta mala.

Eso no está garantizado.

En un sistema operativo, el procesador genera una dirección exacta. Si su página no está residente, el hardware produce un page fault. El sistema sabe con certeza qué página faltó.

En una arquitectura neuronal, el router genera el conjunto de bloques:

$$
\hat A_t = P(x_t,s_{t-1})
$$

Pero existe un conjunto verdadero o ideal:

$$
A_t
$$

Si el router omite un bloque necesario:

$$
A_t \setminus \hat A_t \neq \varnothing
$$

no ocurre automáticamente ningún fallo. El modelo simplemente no accede al bloque. Puede producir una activación incorrecta sin saber que algo faltó.

Esto divide los errores del router en dos casos:

$$
\text{falso positivo} \Rightarrow \text{bytes desperdiciados}
$$

$$
\text{falso negativo} \Rightarrow \text{degradación o repetición del cálculo}
$$

Para convertir un falso negativo en un verdadero page fault necesitaríamos alguno de estos mecanismos:

* un generador exacto y barato de direcciones de bloques;
* un verificador capaz de detectar la omisión antes de continuar;
* un predictor conservador que cargue un superconjunto;
* una ruta densa de respaldo que pueda repetirse desde la capa afectada.

Pero verificar exactamente que una neurona MLP habría sido importante normalmente requiere calcular su activación, lo que exige leer sus pesos. Allí reaparece parte de la circularidad.

Los sistemas existentes resuelven esto de forma aproximada, no como un page fault exacto. LLM in a Flash reportó predictores con falsos negativos y aceptó las omisiones porque las contribuciones eran pequeñas. NeuroPrefetcher reporta aproximadamente 92–96% de retención de precisión densa, no equivalencia exacta. PowerInfer también muestra que reducir demasiado el predictor degrada tanto su precisión como la calidad del modelo. ([arXiv][2])

Por tanto:

> El router circular no desaparece. Se convierte en un problema de **predicción de direcciones semánticas con falsos negativos**.

Es más manejable, pero sigue siendo un problema central.

# 5. El router plano es razonable, pero no porque toda jerarquía implique pointer chasing

Claude también exagera aquí:

> Dominio → capacidad → familia → operador son cuatro cargas dependientes.

Eso sería cierto si se implementara literalmente como cuatro estructuras enlazadas en RAM.

Pero una jerarquía puede representarse mediante:

* índices aritméticos;
* tablas contiguas;
* bitsets;
* árboles almacenados breadth-first;
* clasificación de dos etapas en tensores pequeños;
* un predictor único que emita directamente identificadores de bloques.

Por tanto, la corrección adecuada sería:

> **Eliminar la jerarquía semántica del camino caliente y producir IDs físicos directos.**

No necesariamente eliminar toda selección jerárquica.

Con unos cientos de bloques, una salida plana puede ser adecuada. Con cientos de miles, puntuar todos los bloques cada token puede resultar más caro que una jerarquía bien diseñada.

Además, el predictor no debe darse por insignificante. NeuroPrefetcher utiliza un predictor residente de aproximadamente 206.8 millones de parámetros, unos 414 MB en FP16 y 2.86% del modelo base. Funciona para su escenario de NVMe y GPU, pero claramente no cabe en L2. ([arXiv][1])

Eso demuestra que:

```text
tabla de rutas pequeña
```

y

```text
predictor capaz de seleccionar correctamente
```

no son necesariamente la misma cosa.

# 6. El primer experimento debe medir el operador real, no solamente lecturas

La curva de tamaño de bloque es necesaria, pero no bastan “treinta líneas de C” haciendo lecturas.

El benchmark correcto debe ejecutar una aproximación real del kernel:

```text
seleccionar bloque
→ leer pesos cuantizados
→ descomprimir o dequantizar
→ realizar GEMV/SIMD
→ acumular resultado
```

Debe variar:

| Variable            | Valores importantes                                |
| ------------------- | -------------------------------------------------- |
| Tamaño de bloque    | 64 B hasta varios MB                               |
| Patrón              | secuencial, aleatorio independiente, pointer chase |
| Estado              | caliente en LLC, frío en DRAM                      |
| Paralelismo         | 1 núcleo y todos los núcleos                       |
| Cantidad de bloques | pocos, decenas, cientos                            |
| Layout              | filas separadas, bloques fusionados, coactivación  |
| Formato             | FP16, Q8, Q4 real                                  |
| Páginas             | normales y huge pages                              |
| Reutilización       | inmediata y después de tocar otro working set      |

Intel MLC, por ejemplo, distingue latencia, ancho de banda, accesos aleatorios, accesos secuenciales y latencia bajo carga; esa separación es importante porque un pointer chase serial no representa el mismo comportamiento que varias lecturas independientes capaces de estar simultáneamente en vuelo. ([Intel][4])

La frase:

> 50 bloques equivalen a 50 latencias de DRAM

tampoco es literalmente cierta en todos los casos.

Si cada bloque es de 8 KiB, cada uno contiene muchas líneas de caché y el núcleo puede mantener varias solicitudes pendientes. Se paga una penalización por saltar de bloque, pero no necesariamente 50 latencias completamente serializadas. Eso depende del patrón de dependencia y de la capacidad de memory-level parallelism.

La métrica debe ser directamente:

$$
T_{\text{sparse}}(K,B,\text{layout},\text{threads})
$$

contra:

$$
T_{\text{dense}}(N,\text{threads})
$$

No solamente GB/s teóricos.

# 7. El segundo experimento debe operar sobre bloques, no sobre neuronas

La prueba propuesta por Claude:

```text
top-k neuronas que capturan 90% de la norma
→ unión sobre 32 tokens
```

es un buen primer diagnóstico, pero no decide la arquitectura.

## Problema 1: la norma no es simplemente aditiva

En un SwiGLU, la contribución de la neurona intermedia \(i\) puede aproximarse como:

$$
c_i =
\left[
\operatorname{SiLU}(xW_g)_i
\cdot
(xW_u)_i
\right]W_{d,i}
$$

Ordenar solamente por magnitud de la activación intermedia ignora:

* la norma de la fila correspondiente de \(W_d\);
* alineamientos entre contribuciones;
* cancelaciones;
* efectos sobre capas posteriores.

Debe medirse error de reconstrucción real, no solamente suma de magnitudes.

## Problema 2: neuronas dispersas pueden tocar casi todos los bloques

Podemos descubrir que solo el 10% de las neuronas importa, pero que ese 10% está distribuido sobre el 90% de las páginas físicas.

En ese caso hay dispersión matemática, pero no ahorro de memoria útil.

La prueba debe realizarse para cada tamaño de bloque \(B\):

$$
A_t^{(B)} =
\text{bloques necesarios para mantener el error bajo } \epsilon
$$

Y medir:

$$
U_B(W)=\left|\bigcup_{j=t-W+1}^{t}A_j^{(B)}\right|
$$

No solamente la unión de IDs neuronales.

## Problema 3: la unión no indica supervivencia en caché

Incluso si:

$$
U_B(32)/N = 0.20
$$

ese 20% puede representar 200 MB y no caber en LLC.

Por eso los umbrales de 0.25 y 0.60 no son decisivos. La condición física relevante sería algo parecido a:

$$
\text{working-set bytes entre reutilizaciones}
<
\text{LLC útil}
$$

o, cuando se usa almacenamiento:

$$
\text{working-set activo}
<
\text{búfer DRAM reservado}
$$

La medida correcta es una **curva de fallos contra capacidad residente**, no solo \(U(W)\).

# 8. Falta un tercer experimento

Los dos experimentos de Claude deben complementarse con uno imprescindible:

## Experimento 3: distancia de reutilización bajo el orden real de ejecución

Se registra una traza como:

```text
token
capa
bloque accedido
bytes del bloque
orden temporal
```

Luego se simulan distintas capacidades de caché:

```text
4 MB
8 MB
16 MB
32 MB
64 MB
128 MB
```

y distintas políticas:

```text
LRU
clock
hysteresis contextual
retención predictiva
prefetch del delta
```

El resultado importante es:

$$
M(C)=\text{bytes traídos desde DRAM por token con capacidad }C
$$

Esto responde directamente si la coherencia temporal sobrevive al recorrido completo del modelo.

Puede ocurrir:

```text
U(32) pequeño
pero distancia de reutilización enorme
→ casi cero hits en LLC
```

Ese caso destruiría la afirmación de amortización por ventana aunque la coherencia semántica fuera excelente.

# 9. La fórmula de Claude es un límite ideal, no la ganancia real

Claude propone:

$$
\frac{N W}{U(W)}
\cdot
\frac{BW_{\text{bloque}}}{BW_{\text{secuencial}}}
$$

La fórmula es razonable como límite ideal si:

* cada bloque se carga exactamente una vez por ventana;
* permanece residente;
* no existen fallos de predicción;
* el router es gratuito;
* no hay recomputación;
* el cómputo no domina;
* todos los bloques tienen igual tamaño.

Pero la ecuación real debe incluir más términos:

$$
T_{\text{nuevo}} =
\frac{B_{\text{miss,DRAM}}}{BW_{\text{DRAM}}}
+
\frac{B_{\text{hit,LLC}}}{BW_{\text{LLC}}}
+
\frac{F_{\text{activo}}}{P_{\text{efectivo}}}
+
T_{\text{router}}
+
p_{\text{fallo}}
\left(
T_{\text{carga}}+T_{\text{replay}}
\right)
$$

Y:

$$
\text{speedup}=
\frac{T_{\text{denso}}}{T_{\text{nuevo}}}
$$

El término clave es:

$$
B_{\text{miss,DRAM}}
$$

Ese valor no sale únicamente de \(U(W)\). Sale de la traza de reutilización, la capacidad de caché, el layout y el orden de ejecución.

# Veredicto

La crítica de Claude mejora el documento en tres aspectos decisivos:

* obliga a derivar la granularidad desde el hardware;
* introduce coherencia temporal;
* rechaza la dispersión neuronal arbitrariamente fina.

Pero contiene dos afirmaciones que todavía no están demostradas:

1. **Que mantener el mismo conjunto lógico durante 32 tokens implica leerlo una sola vez desde DRAM.**
2. **Que un error del router se convierte automáticamente en un page fault recuperable.**

Ambas pueden ser falsas.

La tesis corregida debería quedar así:

> **La arquitectura CPU-native debe usar selección espacial en bloques suficientemente grandes y contiguos, colocados según coactivación, con rutas que persistan durante ventanas. Esa persistencia solo cuenta como ahorro cuando reduce movimientos a través de un límite físico real de memoria: NVMe→RAM, RAM→LLC o una carga de pesos amortizada sobre varios tokens mediante prefill, batching, MTP o speculative decoding.**

Y el programa experimental correcto pasa de dos a tres pruebas:

```text
1. Break-even de tamaño de bloque con kernel real.
2. Coherencia temporal a nivel de bloques y error de calidad.
3. Distancia de reutilización y bytes DRAM reales bajo decode autoregresivo.
```

La frase más precisa no es “temporal, no espacial”, sino:

> **gruesa en el espacio, estable en el tiempo y verificablemente residente.**

Esa sí puede convertirse en el contrato de una arquitectura CPU-first.

[1]: https://arxiv.org/html/2608.22643v1 "NeuroPrefetcher: Storage-Aware Sparse LLM Inference via Delta Prefetching"
[2]: https://arxiv.org/html/2312.11514v1 "LLM in a flash:Efficient Large Language Model Inference with Limited Memory"
[3]: https://arxiv.org/html/2605.16786v1 "Lever: Speculative LLM Inference on Smartphones"
[4]: https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html "Intel® Memory Latency Checker v3.12"

### Claude Opus 5

Lo que concedo, sin regatear
La distancia de reutilización es correcta y es la objeción que rompe mi formulación. Si el bloque se usa en la capa 7 del token *t* y no vuelve a tocarse hasta la capa 7 del token *t+1*, entre medias la CPU ha barrido el modelo entero. La distancia de reutilización de cualquier peso en decode autoregresivo es **el modelo activo completo**. Con LLC de 32 MB y ruta activa de cientos de MB, los hits son aproximadamente cero. Reutilización lógica sin residencia física no ahorra un solo byte. "Temporal, no espacial" estaba mal dicho.

Concedo también el falso negativo silencioso, la unión medida sobre bloques y no sobre neuronas, y que el benchmark necesita el kernel real y no lecturas desnudas.

Pero la refutación hereda un supuesto del documento original y no lo toca.

## La grieta

Su sección 3 lista las formas de amortizar una carga de pesos sobre varios tokens: prefill, batching, speculative decoding, MTP, o estado persistente temporal. Todas comparten una condición: **necesitan tokens futuros que aún no existen**. Por eso concluye que en decode interactivo estás atrapado.

Falta un multiplicador. La intensidad aritmética es

$$\text{FLOPs}/\text{byte} = \frac{2 \cdot P\_{\text{activos}}}{B\_{\text{únicos leídos}}}$$

Hay exactamente dos maneras de subir ese cociente: aumentar el numerador reutilizando el peso entre **tokens** (batch, especulación), o reutilizándolo entre **pasos de profundidad dentro del mismo token**. La segunda no requiere conocer el futuro. Está disponible ahora mismo, en el forward pass que ya estás ejecutando.

## La idea: la profundidad como fuente de reutilización

Si los pesos se comparten entre capas (núcleo recursivo, tipo Universal Transformer / ALBERT / recursión con profundidad dinámica), el bloque usado en el paso 3 es literalmente el mismo del paso 11. La distancia de reutilización deja de ser "el modelo entero" y pasa a ser "una iteración del núcleo".

Eso convierte la residencia de propiedad emergente en propiedad de diseño. Ya no mides si sobrevive en caché, lo impones dimensionando el núcleo por debajo del nivel rápido.

Números para ver el salto. Núcleo de 24M parámetros en int4 = 12 MB, iterado 30 veces:

```
Bytes DRAM por token:   12 MB      (una vez, no 30)
FLOPs por token:        2 × 24M × 30 = 1.44 GFLOP
Intensidad:             ~120 FLOPs/byte

```

Contra un denso equivalente en cómputo (720M params int4 = 360 MB por token, intensidad 2 FLOPs/byte). El balance de máquina de una CPU típica está en torno a 7 FLOPs/byte. El denso está 3–4× por debajo, o sea clavado en el techo de memoria. El recursivo está muy por encima: **pasa a ser compute-bound**. Y eso es exactamente lo que quieres, porque a partir de ahí escalas con núcleos, con VNNI, con int8, con todo lo que una CPU sí sabe hacer.

La literatura de looped transformers lo motiva por eficiencia de parámetros o por razonamiento. El argumento de hardware es distinto y creo que no está formulado: **compartir pesos en profundidad es la única forma conocida de subir la intensidad aritmética en batch 1 sin tokens futuros.**

## Afinando el objetivo de residencia

La refutación dice "decenas de MB, LLC". Yo bajaría un nivel más. La L2 privada por núcleo (1–2 MB en silicio reciente) da \~250 GB/s por núcleo, y ocho núcleos suman en agregado un orden de magnitud sobre DRAM, mientras que la LLC compartida es un recurso en disputa.

El objetivo de diseño se vuelve concreto y brutal:

```
slice del núcleo por hilo ≤ L2 privada por núcleo

```

Con 8 hilos × 1.5 MB útiles = 12 MB de núcleo total en int4. Ese es el presupuesto real, y es el que debería fijar la anchura del modelo, no al revés.

## Lo que esto le hace al router

Ahora tienes dos recursos con costes opuestos:

| bytes DRAM FLOPs             |       |       |
| ---------------------------- | ----- | ----- |
| Núcleo recursivo residente   | \~0   | altos |
| Banco de conocimiento en RAM | altos | \~0   |

Razonar es gratis en ancho de banda. Consultar es gratis en cómputo. Dejan de competir por el mismo cuello de botella, que es precisamente lo que el documento quería desacoplar y no sabía cómo.

Y el router deja de tener que elegir entre cientos de miles de bloques. Su decisión primaria es casi binaria:

> ¿este token se resuelve iterando el núcleo, o hace falta traer conocimiento de RAM?

Predecir un bit es incomparablemente más fácil que predecir el conjunto exacto de bloques, y equivocarse es recuperable: si iteras y la incertidumbre no baja, consultas; si consultas de más, gastaste bytes. La circularidad no desaparece, pero se reduce a su versión manejable.

## El falso negativo sí puede convertirse en fallo explícito

Su objeción 4 es correcta como está planteada, pero tiene solución y no exige leer los pesos omitidos. Precomputas, una vez, las normas por fila de $W\_d$ de cada bloque. Son kilobytes, residentes siempre. En runtime, un sketch de bajo rango de la puerta $W\_g$ (también residente) da una puntuación aproximada $\hat{s}\_b$. Entonces

$$\text{error omitido} \le \sum\_{b \notin \hat A\_t} \hat s\_b \cdot |W\_{d,b}|$$

Si esa cota supera $\epsilon$, faltas y cargas. No es exacto porque el sketch no lo es, pero calibrando un margen conservador obtienes un mecanismo con falsos negativos acotados en lugar de silenciosos. Page fault conservador: pagas falsos positivos para eliminar omisiones ciegas.

## El primer experimento cambia

Antes de sus tres pruebas va una que no requiere entrenar nada ni instrumentar ningún modelo, porque decide si el mecanismo existe siquiera en tu silicio:

```
Kernel A: 12 MB int4, iterado 30 veces, slices fijados a L2 por hilo
Kernel B: 360 MB int4, barrido una vez
Mismos FLOPs. Mide tok/s, fallos de LLC, tráfico DRAM real (perf counters).

```

Si A no gana por un factor grande, todo lo demás sobra y el programa muere ahí, barato. Si gana, entonces sí tiene sentido preguntar la parte de ML: si un núcleo de 24M iterado 30 veces puede acercarse en calidad a un denso de 720M. Esa pregunta es cara. La de hardware no lo es, y va primero.

Dos avisos honestos: no puedo verificar NeuroPrefetcher ni las cifras de Lever, son posteriores a lo que conozco, y aceptarlas o no cambia poco el argumento. Y la recursión tiene un coste que no he defendido: si el núcleo es pequeño, la capacidad total baja, y ahí es donde el banco en RAM tiene que hacer todo el trabajo pesado. Ese reparto es la parte que no sé resolver todavía.

### GPT 5.6 Sol

Claude encontró ahora **la grieta arquitectónica correcta**. La reutilización de pesos en profundidad sí es una forma real de aumentar la intensidad aritmética en batch 1 sin conocer tokens futuros.

Pero hay que corregir cuatro puntos antes de convertirlo en el núcleo del documento:

1. Salir del límite de DRAM **no implica automáticamente quedar compute-bound**.
2. El beneficio máximo no es necesariamente \(K\) veces.
3. Treinta iteraciones útiles son un problema de aprendizaje mucho más difícil que treinta lecturas desde L2.
4. El núcleo recursivo no resuelve por sí solo conocimiento, atención ni la proyección al vocabulario.

## 1. La idea central sí funciona físicamente

Sean:

* \(P\): pesos únicos del núcleo.
* \(K\): repeticiones en profundidad.
* \(q\): bytes por peso.
* \(\Pi\): rendimiento efectivo del kernel.
* \(BW_D\): ancho de banda efectivo de DRAM.
* \(BW_{L2}\): ancho de banda efectivo desde L2.

Para un modelo no compartido con \(K\) bloques distintos:

$$
B_{\text{DRAM,denso}}\approx KqP
$$

Para un núcleo compartido que permanece residente:

$$
B_{\text{DRAM,rec}}\approx qP
$$

Por tanto:

$$
AI_{\text{DRAM,rec}}
=
\frac{2KP}{qP}
=
\frac{2K}{q}
$$

En Q4 ideal, \(q=0.5\):

$$
AI_{\text{DRAM,rec}}=4K
$$

Con \(K=30\):

$$
AI_{\text{DRAM,rec}}\approx120\text{ operaciones/byte}
$$

Eso es correcto. El forward sigue haciendo treinta veces el trabajo, pero DRAM solo entrega una copia de los pesos del núcleo, siempre que realmente permanezcan en caché.

Universal Transformer y ALBERT ya demostraron que se pueden compartir pesos entre profundidades; trabajos recientes como Ouro/LoopLM y RecurrentGPT muestran que la recurrencia en profundidad también puede funcionar en modelos generativos modernos. RecurrentGPT, publicado el 18 de agosto de 2026, alcanzó prácticamente el promedio de tareas de un GPT-2 de 774M parámetros usando 288M parámetros únicos y el mismo número aproximado de aplicaciones de bloques. ([arXiv][1])

Por tanto, la hipótesis deja de ser:

> “quizá podamos evitar leer todos los pesos”.

Ahora es:

> **podemos ejecutar mucho más cómputo del que cargamos desde DRAM, reutilizando los mismos pesos durante la profundidad interna del token.**

Ese es un cambio serio.

---

## 2. La intensidad de 120 operaciones/byte solo existe frente a DRAM

Claude concluye demasiado rápido que el núcleo pasa a ser compute-bound.

Aunque los pesos se lean una vez desde DRAM, se siguen leyendo desde L2 en cada iteración:

$$
B_{L2}\approx KqP
$$

Por tanto:

$$
AI_{L2}
=
\frac{2KP}{KqP}
=
\frac{2}{q}
$$

Para Q4:

$$
AI_{L2}\approx4\text{ operaciones/byte}
$$

La arquitectura tendría entonces dos intensidades diferentes:

| Frontera    |   Intensidad aproximada |
| ----------- | ----------------------: |
| DRAM → L2   | \(4K\) operaciones/byte |
| L2 → núcleo |  \(4\) operaciones/byte |

Así que no necesariamente se vuelve compute-bound. Puede quedar:

* limitada por L2;
* limitada por desempaquetado/dequantización Q4;
* limitada por sincronización entre núcleos;
* limitada por instrucciones SIMD;
* limitada por normalizaciones y no linealidades;
* finalmente limitada por cómputo.

La formulación precisa es:

> **La recurrencia residente desplaza el cuello de botella desde DRAM hacia L2, dequantización y ejecución.**

Eso ya sería una victoria enorme, aunque no llegue a compute-bound.

Una aproximación del tiempo sería:

$$
T_{\text{rec}}
\gtrsim
\max\left(
\frac{2KP}{\Pi_{\text{efectivo}}},
\frac{qP}{BW_D}
+
\frac{(K-1)qP}{BW_{L2}}
\right)
+
K\,T_{\text{sync}}
+
T_{\text{otros}}
$$

frente a:

$$
T_{\text{untied}}
\gtrsim
\max\left(
\frac{2KP}{\Pi_{\text{efectivo}}},
\frac{KqP}{BW_D}
\right)
+
K\,T_{\text{sync}}
+
T_{\text{otros}}
$$

En el límite puramente de ancho de banda y con \(K\) grande, la aceleración tiende aproximadamente a:

$$
S_{\max}\approx\frac{BW_{L2}}{BW_D}
$$

no necesariamente a \(K\).

Si L2 entrega ocho veces el ancho de banda sostenible de DRAM, el techo aproximado será ocho veces, no treinta. Después se restan dequantización, barreras y demás costes.

El valor de “7 operaciones/byte” tampoco puede tratarse como balance universal de una CPU. Debe calcularse usando el throughput real del kernel Q4 y el ancho de banda efectivo de esa máquina. Q4 no implica automáticamente dot products nativos de 4 bits; muchos kernels tienen que desempaquetar o convertir antes de usar instrucciones enteras/vectoriales.

---

## 3. El diseño por L2 es viable, pero 12 MB nominales no equivalen a 12 MB utilizables

El planteamiento:

```text
slice del núcleo por núcleo físico ≤ L2 privada
```

es correcto si se cumplen estas condiciones:

* un hilo está fijado permanentemente a cada núcleo físico;
* cada hilo procesa siempre el mismo shard de filas;
* no hay migración del scheduler;
* no se confunden hilos SMT con núcleos físicos;
* los pesos, escalas, metadatos y buffers caben simultáneamente;
* ningún procesamiento externo barre la L2 entre iteraciones;
* las barreras no destruyen el beneficio.

Las L2 privadas no forman automáticamente una caché compartida de 12 MB. Se convierten en una reserva agregada de 12 MB solo si el runtime distribuye físicamente los pesos:

```text
Core 0 → shard 0 siempre
Core 1 → shard 1 siempre
...
Core 7 → shard 7 siempre
```

Esto es compatible con GEMV particionando canales de salida. Un trabajo reciente sobre inferencia cache-resident también distribuye canales de salida entre núcleos y fija los dominios de ejecución, aunque su prototipo utiliza un EPYC 9684X de 96 núcleos y 1.152 GB de LLC por socket, no una pequeña L2 convencional. El trabajo muestra que, una vez residentes los pesos, la sincronización y el movimiento entre niveles de caché pasan a ser los problemas dominantes. ([arXiv][2])

Además:

```text
24M pesos × 0.5 bytes = 12 MB
```

es solo el peso Q4 ideal. Faltan:

* escalas por grupo;
* posibles ceros;
* alineamiento;
* tablas de cuantización;
* estado del kernel;
* activaciones;
* código y stack;
* contaminación por el sistema operativo.

Usar el 100% de la capacidad nominal es demasiado agresivo. El presupuesto inicial debería ser quizá 60–75% de la L2 físicamente disponible, determinado por medición. En el ejemplo de ocho núcleos con 1.5 MB útiles nominales, empezaría probando núcleos de aproximadamente 7–9 MB reales, no directamente 12 MB.

En una VPS tampoco puede suponerse:

```text
8 vCPU = 8 núcleos físicos independientes
```

ni que los contadores de memoria, el pinning o la L2 permanezcan bajo control. El gate decisivo debería ejecutarse en bare metal.

---

## 4. El experimento A/B debe aislar solamente el peso compartido

La comparación propuesta por Claude es buena en intención, pero “un bloque de 12 MB repetido contra un bloque de 360 MB barrido una vez” puede introducir diferencias de forma matricial y eficiencia del kernel.

La comparación limpia sería:

### A — profundidad recurrente

```text
30 aplicaciones
mismos pesos de 12 MB
mismas formas matriciales
estado h actualizado entre pasos
```

### B — profundidad convencional

```text
30 aplicaciones
30 copias distintas de pesos de 12 MB
mismas formas matriciales
mismo estado h actualizado entre pasos
```

Ambas variantes deben ejecutar exactamente:

* los mismos GEMV;
* RMSNorm;
* SwiGLU;
* residuales;
* dequantización;
* sincronizaciones;
* dimensiones;
* cantidad de operaciones.

La única diferencia será:

```text
A: dirección de pesos repetida
B: dirección de pesos distinta por profundidad
```

Añadiría una tercera condición:

### C — control de expulsión

Igual que A, pero tocando deliberadamente un buffer grande entre iteraciones para expulsar el núcleo de L2.

Entonces:

```text
A ≫ C ≈ B
```

confirmaría que la ganancia procede realmente de residencia y no de otro detalle del kernel.

El barrido mínimo debería variar:

$$
K\in\{1,2,4,8,16,32\}
$$

y:

$$
Pq\in
\{0.25,0.5,0.75,1.0,1.25,2.0\}
\times
\text{L2 útil agregada}
$$

Las métricas decisivas serían:

* bytes leídos por los controladores de memoria;
* L2 hits y fills;
* LLC misses;
* ciclos por aplicación;
* tiempo por forward;
* escalado con núcleos;
* coste de cada barrera;
* throughput real de MAC del kernel;
* cambio abrupto al cruzar la capacidad útil de L2.

El gate de hardware pasa si aparece una rodilla clara:

```text
debajo de L2 → tráfico DRAM próximo a una copia del núcleo
encima de L2 → tráfico DRAM crece hacia K copias
```

Yo exigiría al menos una aceleración de aproximadamente \(2\times\) en el kernel completo para justificar el siguiente trabajo de ML; \(3\times\) o más sería una señal fuerte. No exigiría \(30\times\), porque ese no es el techo físico correcto.

---

## 5. Hay dos barridos externos que pueden destruir la residencia

Aunque el núcleo recurrente quepa en L2, un modelo de lenguaje todavía tiene dos componentes grandes.

### Proyección de salida

Una cabeza convencional de:

$$
32\,000\times768
$$

contiene:

$$
24.6\text{ millones de pesos}
$$

Es decir:

$$
12.3\text{ MB en Q4 ideal}
$$

La cabeza por sí sola ya consumiría todo el presupuesto del ejemplo.

En Ouro, el vocabulario es 49,152 y el estado oculto tiene dimensión 2,048. La matriz vocabulario–estado contiene aproximadamente 101 millones de pesos: unos 50 MB en Q4 ideal, incluso si entrada y salida comparten pesos. ([arXiv][3])

Esto no invalida la arquitectura, pero obliga a rediseñar también la salida. Trabajos recientes encuentran que la cabeza puede representar entre 20% y 60% de los parámetros en modelos pequeños, y han explorado clustering o subvocabularios para no escanear toda la matriz en cada token. FlashHead reporta aceleraciones end-to-end de hasta \(1.75\times\) en algunos modelos pequeños; CSV-Decode propone cotas geométricas para certificar que los tokens omitidos no pueden entrar en el top-\(k\). ([arXiv][4])

Por tanto, una arquitectura L2-native necesita algo como:

```text
cabeza jerárquica
o
subvocabulario certificado
o
tokenización pequeña
o
proyección de salida factorizada
```

No puede dejar intacto el LM head convencional.

### Atención y KV cache

Si el núcleo recursivo ejecuta atención completa en cada una de las 30 vueltas, puede leer el KV cache treinta veces. Entonces se elimina el tráfico de pesos, pero se multiplica el tráfico de contexto.

Los propios modelos looped han tenido que abordar este problema. Ouro estudia compartir el KV cache entre vueltas; MELT mantiene un solo KV actualizado entre loops; LT2 sustituye atención completa por mixers lineales, dispersos o híbridos precisamente porque repetir atención completa limita la eficiencia. ([arXiv][3])

La forma correcta sería:

> **leer contexto y memoria una vez, condensarlos en un scratchpad pequeño y después iterar localmente sin volver a recorrer el contexto.**

---

## 6. La literatura ya responde parcialmente al problema de capacidad

Claude termina diciendo que no sabe cómo repartir trabajo entre el núcleo pequeño y el banco en RAM.

Aquí existe un resultado muy relevante: Ouro/LoopLM midió por separado almacenamiento de conocimiento y manipulación de conocimiento.

El resultado fue:

* repetir el núcleo no aumentó la cantidad de hechos memorizables;
* modelos looped y no-looped conservaron aproximadamente la misma capacidad por parámetro;
* la recurrencia mejoró notablemente la composición y manipulación del conocimiento existente.

El trabajo resume la capacidad factual en aproximadamente dos bits por parámetro en su prueba sintética y concluye que incrementar las vueltas no incrementa la capacidad de almacenamiento factual. ([arXiv][3])

Eso significa:

$$
24M\text{ parámetros repetidos 30 veces}
\neq
720M\text{ parámetros de conocimiento}
$$

Las treinta vueltas pueden proporcionar capacidad computacional, pero el núcleo sigue almacenando aproximadamente 24M parámetros únicos de conocimiento.

Esto vuelve obligatorio el reparto:

```text
Núcleo recurrente:
    algoritmos
    composición
    sintaxis
    control
    planificación
    transformación de estado

Banco en RAM:
    hechos
    entidades
    relaciones
    conocimiento especializado
    episodios
    adaptadores de dominio
```

Los resultados modernos apoyan reducciones aproximadas de 2–3 veces en parámetros únicos, no 30 veces. Ouro reporta modelos de 1.4B y 2.6B comparables en varias pruebas con modelos de 4B y 8B; RecurrentGPT obtiene paridad promedio aproximada usando 37% de los parámetros del baseline iso-FLOPs. ([arXiv][3])

Por tanto, el ejemplo \(24M\times30\) es válido como prueba de hardware, pero no debe presentarse como una equivalencia de calidad probable con 720M. El barrido de ML debe estudiar relaciones:

$$
2\times,\;4\times,\;8\times,\;16\times,\;30\times
$$

y medir dónde colapsa la capacidad.

---

## 7. Existe una tensión entre lo que quiere el hardware y lo que permite el aprendizaje

El hardware quiere \(K\) grande:

$$
AI_{\text{DRAM}}\propto K
$$

Pero los modelos recursivos actuales no continúan mejorando indefinidamente.

Ouro fue entrenado con cuatro vueltas; normalmente alcanzó su mejor calidad alrededor de cuatro y después degradó entre cinco y ocho. RecurrentGPT resolvió aproximadamente 95% de la divergencia respecto a su estado final en las primeras cuatro vueltas; extender más allá de las seis vueltas de entrenamiento empeoró ligeramente la pérdida. LoopFormer señala que la repetición ingenua suele converger hacia estados estancados. ([arXiv][3])

Esto crea el verdadero conflicto:

```text
Hardware:
    quiere 16–32 reutilizaciones

Dinámica neuronal:
    puede converger o estancarse en 4–8
```

Un único bloque aplicado ciegamente treinta veces probablemente:

* converge demasiado pronto;
* repite trabajo;
* oscila;
* acumula error de cuantización;
* pierde especialización por profundidad.

RecurrentGPT muestra que la recurrencia simple empeora la pérdida; necesita prelude, coda, un ancla fija, proyección recurrente y una puerta de escritura. LoopQ muestra además que el error de cuantización se acumula a través de vueltas compartidas, por lo que Q4 recurrente no puede tratarse como una cuantización ordinaria. ([arXiv][5])

La arquitectura necesitará diferenciación de paso barata:

$$
h_{r+1}
=
h_r
+
g_r(h_r,a)
\odot
C_\theta(h_r,a,\tau_r)
$$

donde:

* \(C_\theta\) contiene los pesos grandes compartidos;
* \(a\) es un ancla o scratchpad fijo;
* \(\tau_r\) indica la profundidad;
* \(g_r\) decide cuánto escribir;
* los parámetros dependientes de \(r\) deben ser diminutos y residentes.

Así se conserva el beneficio de compartir casi todos los pesos sin exigir que cada vuelta represente exactamente la misma transformación.

---

## 8. El router binario simplifica el problema, pero no lo elimina

La pregunta:

> “¿puedo continuar pensando o necesito consultar RAM?”

es mucho más fácil que predecir directamente miles de microexpertos. Eso es correcto.

Pero después de responder “consultar”, todavía queda:

> “¿qué dirección o página de RAM debo consultar?”

El router detallado no desaparece; se ejecuta con menor frecuencia y después de que el núcleo haya construido una consulta mejor.

La secuencia adecuada sería:

```text
READ
  El núcleo formula una consulta.
  Se recuperan pocas páginas de RAM.
  Se comprimen en un scratchpad.

THINK
  El núcleo L2-residente itera K veces.
  No toca el banco ni el contexto grande.

VERIFY
  Evalúa progreso, contradicción y cobertura.

READ AGAIN
  Solo cuando falta información.

WRITE
  Produce el token y actualiza estado persistente.
```

Esto convierte el sistema en una arquitectura de **lectura escasa y pensamiento intensivo**, no en un MoE de microoperadores.

La frase correcta tampoco es:

```text
razonar es gratis en ancho de banda
consultar es gratis en cómputo
```

Sino:

```text
razonar consume principalmente L2 + cómputo
consultar consume principalmente DRAM + búsqueda
```

Los dos caminos usan recursos diferentes y dejan de golpear simultáneamente el mismo cuello de botella.

---

## 9. La cota del falso negativo propuesta por Claude todavía no es válida

Para un bloque SwiGLU, la contribución tiene aproximadamente esta forma:

$$
y_b
=
W_{d,b}
\left[
\operatorname{SiLU}(W_{g,b}h)
\odot
(W_{u,b}h)
\right]
$$

Una cota válida comienza por:

$$
\|y_b\|_2
\le
\|W_{d,b}\|_2
\cdot
\left\|
\operatorname{SiLU}(W_{g,b}h)
\odot
(W_{u,b}h)
\right\|_2
$$

No basta con:

* un sketch de \(W_g\);
* una norma por fila de \(W_d\).

También se necesita:

* una cota de \(W_u h\);
* el error certificado del sketch;
* la interacción multiplicativa de SwiGLU;
* cancelaciones entre bloques;
* amplificación posterior del error.

En recurrencia:

$$
\|\delta h_K\|
\lesssim
\sum_{r=1}^{K}
\left(
\prod_{j=r+1}^{K}\|J_j\|
\right)
\|e_r\|
$$

donde \(e_r\) es el error introducido en la vuelta \(r\) y \(J_j\) el Jacobiano de las vueltas posteriores.

Aunque una omisión sea pequeña localmente, puede amplificarse durante treinta pasos. LoopQ observa precisamente propagación y acumulación de errores de cuantización entre vueltas. ([arXiv][6])

Sí puede construirse un mecanismo conservador, pero necesitaría:

$$
s_b
\le
\hat s_b+\delta_b
$$

con \(\delta_b\) certificado, no solamente calibrado empíricamente. Cuanto más conservador sea, más páginas falsas cargará.

Las cotas geométricas certificadas parecen especialmente prometedoras para la cabeza de vocabulario, donde la operación es lineal y CSV-Decode ya demuestra el principio. Aplicarlas dentro de una SwiGLU recursiva es considerablemente más difícil. ([arXiv][4])

---

# La arquitectura que queda después de todas las correcciones

Ya no sería el sistema de microexpertos del documento original.

Sería:

```text
┌───────────────────────────────────────────────┐
│ DRAM: memoria semántica paginada              │
│ hechos, entidades, relaciones, adapters       │
└──────────────────────┬────────────────────────┘
                       │ lecturas infrecuentes
                       ▼
┌───────────────────────────────────────────────┐
│ LLC/L1: scratchpad contextual                 │
│ evidencia recuperada, estado, índices, KV     │
└──────────────────────┬────────────────────────┘
                       │ estado compacto
                       ▼
┌───────────────────────────────────────────────┐
│ L2 privadas: núcleo procedural compartido     │
│ mismo bloque, K iteraciones, hilos fijados    │
│ puertas y modulación ligera por profundidad   │
└──────────────────────┬────────────────────────┘
                       ▼
┌───────────────────────────────────────────────┐
│ Cabeza de salida selectiva/certificada        │
└───────────────────────────────────────────────┘
```

Matemáticamente:

$$
m_t=
\operatorname{Retrieve}(M,q(h_t))
$$

$$
z_t^{(0)}
=
\operatorname{Prelude}(h_t,m_t,s_{t-1})
$$

$$
z_t^{(r+1)}
=
C_\theta
\left(
z_t^{(r)},
m_t,
z_t^{(0)},
\tau_r
\right),
\qquad r=0,\ldots,K_t-1
$$

$$
\text{si progreso insuficiente}
\Rightarrow
\operatorname{RetrieveAgain}
$$

$$
p(x_{t+1})
=
\operatorname{SelectiveHead}
\left(
z_t^{(K_t)}
\right)
$$

La separación fundamental sería:

$$
\boxed{
\text{capacidad de cómputo}
\;\not=\;
\text{capacidad de conocimiento}
\;\not=\;
\text{bytes DRAM por token}
}
$$

* La profundidad recurrente suministra cómputo.
* El banco suministra capacidad factual.
* La residencia en caché elimina tráfico repetido.
* El halting suministra profundidad dinámica.
* La cabeza selectiva evita que el vocabulario vuelva a dominar.
* El scratchpad evita releer el contexto en cada vuelta.

# Veredicto

Claude ha identificado un mecanismo físico válido que el documento anterior no tenía:

> **reutilizar pesos dentro de la profundidad del mismo token es la forma más directa de elevar la intensidad frente a DRAM en batch 1 sin depender de tokens futuros.**

No afirmaría que es literalmente la única forma posible, porque también existen factorizaciones, pesos implícitos y operadores estructurados. Pero sí es la forma general más limpia y mejor alineada con un modelo neuronal profundo.

La tesis central del documento debería cambiar a:

> **Un modelo CPU-native debe usar un núcleo recurrente suficientemente pequeño para permanecer en caché, ejecutar sobre él profundidad dinámica y obtener su capacidad de conocimiento mediante lecturas paginadas e infrecuentes desde RAM.**

Lo que queda sin demostrar ya no es el mecanismo de hardware. Ese mecanismo es plausible, medible y cuenta con apoyo indirecto de arquitecturas recurrentes y sistemas cache-resident. Lo realmente abierto es:

$$
\boxed{
\text{¿cuánta calidad puede conservar un núcleo procedural pequeño
cuando el conocimiento se externaliza?}
}
$$

Y el primer gate correcto no requiere entrenar nada: **mismo grafo, mismas operaciones, pesos compartidos residentes contra pesos distintos no residentes**. Si ahí no aparece una ventaja clara, se mata la línea. Si aparece, entonces la investigación de ML queda justificada.

[1]: https://arxiv.org/html/1807.03819v1 "Universal Transformers"
[2]: https://arxiv.org/html/2606.25353v1 "Cache-Resident LLM Inference in GB-Scale Last-Level Caches"
[3]: https://arxiv.org/html/2510.25741v2 "Scaling Latent Reasoning via Looped Language Models"
[4]: https://arxiv.org/html/2511.21702 "CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference"
[5]: https://arxiv.org/html/2608.15062v2 "RecurrentGPT: Expressive Depth through Recurrent Modulation in Transformers"
[6]: https://arxiv.org/pdf/2605.16343 "LoopQ: Quantization for Recursive Transformers"

### Claude Opus 5

Lo que concedo, en firme
La pared de L2 es real y mi "pasa a ser compute-bound" era falso. Un GEMV lee cada peso una vez y hace dos operaciones con él: la intensidad aritmética es 2/q **en cualquier frontera**, la recursión no la toca. Concedo también la cota SwiGLU (acoté la puerta e ignoré el término multiplicativo y la amplificación por el jacobiano a lo largo de K), el diseño A/B/C con control de expulsión, que es mejor que el mío, y el resultado de capacidad: repetir no almacena.

Ahora, dos correcciones y una salida.

## El techo no es 8×

`S_max ≈ BW_L2/BW_DRAM` es la forma correcta, pero el numerador está mal contado. La L2 es **privada y escala con núcleos**; la DRAM es compartida y fija. Con 32–64 B/ciclo por núcleo a \~4 GHz son 128–256 GB/s por núcleo. Ocho núcleos fijados a sus shards agregan 1–2 TB/s contra 50–70 GB/s de DRAM sostenida: **20–40×, y crece linealmente con el recuento de núcleos**.

Eso es lo que hace la línea interesante en CPU y no en GPU. En una GPU la HBM ya te da varios TB/s y el margen de residencia es estrecho. En una CPU la brecha DRAM↔L2 agregada es de casi dos órdenes de magnitud, y es exactamente donde una arquitectura recurrente residente vive.

## La objeción de AI=2/q no se concede: se disuelve

`AI = 2/q` sale de un supuesto que ningún documento ha cuestionado: **que el estado es un vector**.

Con VNNI, un núcleo hace \~256 ops/ciclo y recibe \~64 B/ciclo de L2. El balance de máquina en la frontera L2 es 4 ops/byte. Un GEMV Q4 tiene exactamente 4. Estás clavado en la rodilla, y cualquier coste de dequantización, RMSNorm o barrera te empuja por debajo del techo de memoria. GPT tiene razón en eso, y esa coincidencia numérica no es casual: es la razón de que la inferencia batch 1 sea estructuralmente ineficiente.

Si el estado es una **matriz de m slots** en lugar de un vector, cada iteración pasa de GEMV a GEMM de m columnas:

$$AI\_{L2} = \frac{2Km,P}{Kq,P} = \frac{2m}{q}, \qquad AI\_{DRAM} = \frac{2Km}{q}$$

Con m=16 y Q4: 64 ops/byte contra un balance de 4. Dieciséis veces por encima del techo.

Seamos precisos sobre qué compra eso, porque no son FLOPs gratis. Pasado el punto de balance, m cuesta tiempo proporcionalmente. Lo que compra es otra cosa, y es mucho:

- **La dequantización se amortiza entre los m slots.** Desempaquetas el bloque Q4 una vez y lo usas m veces. Ese coste que GPT señala como posible cuello desaparece por división.
- **Recuperas el pico que el GEMV no puede alcanzar.** Un GEMV corre al 10–30% del pico por bloqueo de registros y latencia de reducción; un GEMM con m=16 llega al 60–90%. Son 3–5× de reloj de pared por FLOP, y son reales.
- **Margen sobre el techo.** Con AI=4 estás en la rodilla y toda barrera te hunde. Con AI=64 las sincronizaciones y normalizaciones dejan de decidir.

## Esto disuelve la tensión de su sección 7

GPT identifica el conflicto: el hardware quiere K=16–32, la dinámica se estanca en 4–8. Con m, el conflicto desaparece porque **m y K son sustitutos para la intensidad aritmética pero no para la dinámica de aprendizaje**.

A FLOPs iguales, `K·m = const`. Si el modelo deja de mejorar pasadas seis vueltas, gastas el resto del presupuesto en anchura de estado en vez de en profundidad. Misma intensidad, mismo coste, mejor dinámica: un workspace de 16 slots tiene mucho más espacio para no colapsar en un punto fijo que un vector iterado treinta veces, que es precisamente el estancamiento que reportan.

Y el workspace resuelve sus secciones 5 y 8 sin maquinaria adicional. El contexto y las páginas recuperadas de RAM se condensan **en los slots**, una vez. Después el núcleo itera localmente sin volver a tocar el KV ni el banco. La atención slot-a-slot es m×m, o sea nada. El scratchpad que GPT pide no es un componente extra: es el estado.

El núcleo queda como un transformer diminuto sobre m tokens latentes con pesos compartidos entre K vueltas, con roles fijos por slot para evitar que colapsen entre sí. Es una arquitectura estándar y entrenable, lo cual importa: la sección 7 de GPT es una advertencia sobre recurrencia ingenua, y esto no lo es.

Adyacencias, declaradas: el array latente de Perceiver, los register tokens de ViT, y sobre todo el estado matricial de Mamba-2/SSD, que existe por esta misma razón de intensidad aritmética, solo que motivada por tensor cores. Lo que no he visto es esta combinación concreta: estado matricial **para permanecer en L2 y amortizar dequantización en un núcleo de pesos compartidos en CPU**.

## Donde GPT exagera: la cabeza

32000×768 en Q4 son 12 MB. A 60 GB/s eso es 200 µs, o sea un techo de \~5000 tok/s **solo por la cabeza**. Eso no es un problema, es un no-problema disfrazado por comparar contra un núcleo hipotéticamente más rápido.

Haz el presupuesto al revés. Objetivo realista de 50 tok/s en un desktop: 1.2 GB por token de DRAM. La cabeza consume 12–50 MB, es decir el 2–4%.

De ahí sale la conclusión que ninguno de los dos documentos extrae:

> **La recursión residente no resuelve el problema de bytes/token. Lo aísla.** Retira el cómputo de la ecuación de memoria y deja prácticamente el presupuesto entero de DRAM disponible para una sola cosa: traer conocimiento.

Eso cambia la naturaleza de lo que queda abierto. Ya no es "cómo enruto entre microexpertos manteniendo la calidad". Es "con 1 GB por token y cero presión de cómputo, ¿cuánto conocimiento útil puedo traer?". Y eso es recuperación pura: ANN, cuantización de producto, layouts por coactivación, con recall certificable, sin la circularidad del router porque **la consulta se formula después de que el núcleo haya pensado**, no antes. La certificación que GPT exige correctamente pasa de tener que acotar cada bloque SwiGLU de cada capa a acotar un top-k lineal una o dos veces por token, que es donde CSV-Decode ya funciona.

## El gate, corregido

Su A/B/C está bien. Le añadiría una tercera dimensión al barrido, porque es la que decide:

```
m ∈ {1, 4, 8, 16, 32}   con K·m fijado a FLOPs constantes

```

Y una métrica que su lista no incluye: **fracción del pico de MAC efectivamente alcanzada**. Si m=16 no sube esa fracción de \~0.2 a \~0.6, mi argumento de GEMM se cae y solo queda la amortización de dequant.

Predicción falsable, para que quede en el ledger: la ganancia sobre el kernel completo será superlineal al pasar de m=1 a m=8 a FLOPs constantes, y luego se aplanará. Si la curva es plana desde m=1, me equivoqué y el estado matricial no aporta nada aquí.

Sobre las referencias que citas: varias son posteriores a lo que conozco y no las he verificado. Ouro, LoopQ y el trabajo cache-resident, si dicen lo que dices, apuntan todos en esta dirección, pero conviene que compruebes que existen antes de construir encima.

### GPT 5.6 Sol

Esta vez Claude identificó **la pieza que realmente faltaba**: no basta con hacer recurrente y residente el núcleo; hay que darle un **estado matricial con varios slots**, para que la operación central deje de ser GEMV y pase a ser un GEMM pequeño dentro de un solo token.

Eso cambia sustancialmente la arquitectura. Pero todavía hay dos exageraciones:

1. \(m\) y \(K\) no son sustitutos completos; solo lo son para algunas métricas físicas.
2. Certificar el top-\(k\) de una búsqueda no certifica que el conocimiento recuperado sea suficiente.

## 1. La corrección de \(m\) slots es válida

Con un estado vectorial:

$$
x\in\mathbb{R}^{d}
$$

la operación es:

$$
y=Wx
$$

Cada peso se usa una vez:

$$
AI_{\text{pesos}}\approx\frac{2}{q}
$$

Con Q4 ideal, \(q=0.5\):

$$
AI\approx4\text{ operaciones/byte}
$$

Pero si el estado es:

$$
X\in\mathbb{R}^{d\times m}
$$

entonces:

$$
Y=WX
$$

y cada peso se reutiliza para los \(m\) slots:

$$
AI_{L2}\approx\frac{2m}{q}
$$

Mientras que, si los pesos permanecen residentes durante \(K\) iteraciones:

$$
AI_{\text{DRAM}}\approx\frac{2Km}{q}
$$

Por tanto, con \(m=16\) y Q4:

$$
AI_{L2}\approx64
$$

Ese es un cambio físico real. La arquitectura deja de depender de un kernel GEMV de batch 1 y crea un **batch interno dentro del estado cognitivo**.

La fórmula más exacta, incluyendo activaciones, sería:

$$
AI_{L2}
=
\frac{2d_od_im}
{
q\,d_od_i+
s\,m(d_i+d_o)+B_{\text{intermedios}}
}
$$

La aproximación \(2m/q\) solo vale cuando el tráfico de pesos domina, pero para dimensiones suficientemente grandes y \(m\) moderado es una aproximación razonable.

También es correcta la amortización de dequantización: un microkernel bien fusionado puede cargar y desempaquetar un tile Q4 una vez, y multiplicarlo por varios vectores de slots antes de descartarlo. La dequantización no desaparece literalmente, pero su coste por MAC puede caer aproximadamente con \(m\) hasta que empiecen a dominar registros, activaciones y otras operaciones.

## 2. El ancho de banda L2 debe agregarse, pero 20–40× no es el speedup automático

Claude corrige correctamente que, con shards fijados permanentemente a núcleos físicos:

$$
BW_{L2,\Sigma}
=
\sum_i BW_{L2,i}
$$

La comparación correcta es contra el ancho de banda compartido de DRAM:

$$
r=\frac{BW_{L2,\Sigma}}{BW_{\text{DRAM}}}
$$

Por tanto, en un desktop con pocos canales de memoria, un cociente teórico de decenas es plausible. Mi ejemplo anterior de aproximadamente \(8\times\) era demasiado conservador si estaba usando una L2 por núcleo contra DRAM total.

Pero la aceleración de residencia para \(K\) vueltas no es directamente \(r\). Ignorando cómputo:

$$
T_{\text{untied}}
\approx
\frac{KqP}{BW_D}
$$

$$
T_{\text{tied}}
\approx
\frac{qP}{BW_D}
+
\frac{(K-1)qP}{BW_{L2,\Sigma}}
$$

Luego:

$$
S_{\text{res}}
=
\frac{K}
{1+\frac{K-1}{r}}
$$

Con \(r=30\):

| \(K\) | Speedup máximo por residencia |
| ----: | ----------------------------: |
|     2 |                \(1.94\times\) |
|     4 |                \(3.64\times\) |
|     6 |                \(5.14\times\) |
|     8 |                \(6.49\times\) |
|    32 |               \(15.74\times\) |

Así que la gran brecha L2↔DRAM importa, pero para una recurrencia de cuatro o seis pasos el techo continúa siendo aproximadamente cuatro o seis veces. Además, 32–64 bytes/ciclo puede representar un techo microarquitectónico, no lo que sostendrá un kernel Q4 completo con escalas, dequantización, activaciones y barreras.

La cifra decisiva sigue siendo:

$$
BW_{L2,\Sigma}^{\text{kernel real}}
$$

no el ancho de banda nominal del camino de carga.

## 3. \(m\) y \(K\) no son equivalentes

Claude dice que \(m\) y \(K\) son sustitutos para intensidad. Eso es cierto frente a DRAM, pero no en general.

| Propiedad                    | Dependencia principal |
| ---------------------------- | --------------------: |
| Intensidad frente a DRAM     |                \(Km\) |
| Intensidad frente a L2       |                 \(m\) |
| Reutilización por residencia |                 \(K\) |
| Profundidad secuencial       |                 \(K\) |
| Tamaño del workspace         |                \(md\) |
| Atención entre slots         |             \(Km^2d\) |
| Número de barreras           |                 \(K\) |

Por tanto:

$$
m\uparrow
$$

mejora el kernel, amplía el workspace y reduce tráfico L2 por MAC.

Pero:

$$
K\downarrow
$$

reduce la cantidad de transformaciones secuenciales que pueden componerse.

A FLOPs constantes:

$$
C=Km
$$

el tiempo aproximado del núcleo residente sería:

$$
T(m)
\gtrsim
\frac{qP}{BW_D}
+
\frac{\left(\frac{C}{m}-1\right)qP}{BW_{L2,\Sigma}}
+
\frac{2PC}{\eta(m)\Pi_{\text{pico}}}
+
\frac{C}{m}T_{\text{sync}}
+
T_{\text{slots}}(m)
$$

Aquí:

* el primer término es fijo;
* el tráfico L2 cae aproximadamente como \(1/m\);
* las barreras caen como \(1/m\);
* la eficiencia \(\eta(m)\) debería mejorar al pasar de GEMV a small-GEMM;
* la interacción entre slots aumenta con \(m\);
* la profundidad secuencial disminuye.

La tensión no desaparece. Se convierte en una optimización bidimensional:

> **suficientes slots para que el hardware trabaje bien, suficientes vueltas para que el modelo componga operaciones.**

Un punto inicial razonable para investigar sería algo como:

$$
m\in\{4,8,16\},\qquad K\in\{4,6,8\}
$$

No porque sepamos que sea óptimo, sino porque evita ambos extremos:

```text
m = 1, K = 32:
buen reuso temporal, GEMV deficiente, riesgo de convergencia.

m = 32, K = 1:
GEMM eficiente, pero prácticamente ninguna profundidad recurrente.
```

Además, con \(Km=32\), escoger \(m=16\) obliga a \(K=2\). No se puede conservar simultáneamente una profundidad de seis vueltas y dieciséis slots sin aumentar el presupuesto a:

$$
Km=96
$$

## 4. El estado matricial y la recursión son dos ganancias diferentes

Hay que evitar atribuir todo el beneficio a la recursión.

Un modelo no recurrente que procese \(m\) slots también convierte sus proyecciones en GEMM. Por eso la comparación correcta debe mantener el mismo \(m\) en las variantes tied y untied:

```text
A: m slots, K vueltas, pesos compartidos y residentes.
B: m slots, K capas con pesos diferentes.
C: igual que A, pero expulsando los pesos entre vueltas.
```

Entonces:

* \(A\) contra \(B\) mide residencia y weight sharing;
* \(A\) contra \(C\) confirma que la ganancia procede de caché;
* \(m=1\) contra \(m>1\) mide matrixización y amortización de dequant;
* mantener \(Km\) constante mide el intercambio arquitectónico profundidad↔workspace.

Sin esa separación, una mejora obtenida por small-GEMM podría atribuirse erróneamente al núcleo recurrente.

## 5. El workspace latente es una dirección sólida

La estructura propuesta tiene una adyacencia muy clara con Perceiver:

* array latente pequeño;
* compresión del input mediante cross-attention;
* procesamiento profundo dentro del espacio latente;
* posibilidad de compartir pesos entre aplicaciones del núcleo.

Perceiver incluso describe su variante compartida como una recurrencia desenrollada en profundidad. Pero también encuentra que volver a consultar el input puede mejorar calidad porque un cuello de botella latente único puede dejar información fuera. Eso significa que los slots sí pueden reemplazar al scratchpad separado, pero probablemente no eliminan completamente el mecanismo `READ AGAIN`. ([arXiv][1])

Mamba-2 también respalda parcialmente la intuición física: su rediseño convierte más trabajo de estado en multiplicaciones matriciales y permite aumentar la dimensión de estado de aproximadamente 16 a 64–128 en sus experimentos. Sin embargo, su aceleración principal está formulada para procesamiento por chunks y tensor cores; no demuestra directamente el caso específico de decode CPU batch 1 con pesos residentes en L2. ([Tri Dao][2])

La arquitectura debería asignar papeles explícitos a los slots, por ejemplo:

```text
1–2 slots de control y consulta
4–8 slots de evidencia recuperada
4–8 slots de trabajo intermedio
1–2 slots de salida y verificación
```

Con embeddings de rol, puertas de escritura y algún mecanismo que impida que todos converjan a la misma representación.

## 6. Claude tiene razón sobre la cabeza de vocabulario

Mi objeción anterior exageró su importancia para un objetivo de 50 tokens/s.

Para:

$$
V=32\,000,\qquad d=768
$$

la cabeza tiene:

$$
Vd=24.6M\text{ pesos}
$$

En Q4 ideal:

$$
12.3\text{ MB}
$$

A 60 GB/s:

$$
\frac{12.3\text{ MB}}{60\text{ GB/s}}
\approx0.205\text{ ms}
$$

A 50 tokens/s hay:

$$
20\text{ ms/token}
$$

Por tanto, esa cabeza consume aproximadamente el 1% del presupuesto temporal ideal. Incluso una cabeza de 50 MB rondaría 0.83 ms, alrededor del 4%.

Conclusión corregida:

> **Para una meta inicial de 50 tokens/s, no hace falta rediseñar primero la cabeza.**

Una cabeza selectiva empieza a importar cuando todo el resto del modelo baja a unos pocos cientos de microsegundos o cuando el vocabulario y la dimensión crecen mucho más.

## 7. “Un gigabyte de conocimiento por token” es un presupuesto, no todavía una solución

La inversión del presupuesto es válida:

$$
60\text{ GB/s}/50\text{ tok/s}
=
1.2\text{ GB/token}
$$

En la práctica no se puede consumir el 100% de DRAM exclusivamente en conocimiento. Hay que reservar tráfico para:

* la primera carga del núcleo;
* la cabeza;
* activaciones;
* índice ANN;
* escrituras;
* memoria del sistema;
* accesos aleatorios que no alcanzan ancho de banda secuencial.

Pero el orden de magnitud puede seguir siendo cientos de megabytes por token. Eso es muchísimo más margen que el que tendría un Transformer denso que ya consume casi todo el ancho de banda leyendo pesos.

Lo que no es correcto es llamar a eso “cero presión de cómputo”. Por ejemplo:

$$
P=24M,\quad K=6,\quad m=16
$$

produce:

$$
2PKm
=
4.608\text{ GFLOP/token}
$$

A 50 tokens/s:

$$
230.4\text{ GOP/s}
$$

Eso puede ser viable en una CPU moderna con kernels enteros eficientes, pero no es gratis. Y todavía falta integrar la evidencia recuperada.

## 8. CSV-Decode no certifica recuperación de conocimiento

Comprobé la referencia: CSV-Decode existe y utiliza cotas geométricas para certificar el top-\(k\) exacto o una aproximación acotada de softmax en la **proyección lineal del vocabulario**. Allí el problema está perfectamente definido:

$$
\operatorname{arg\,topk}_i\; h^\top e_i
$$

y puede demostrarse que ciertos embeddings no pueden superar el umbral actual. ([arXiv][3])

En un banco semántico, un índice ANN puede certificar:

> “Estos son los vecinos más cercanos bajo esta métrica.”

Pero no puede certificar automáticamente:

> “Estos vecinos contienen todo el conocimiento necesario para responder correctamente.”

Son garantías diferentes.

El núcleo puede formular una consulta incorrecta, usar una representación insuficiente o no saber que existe una excepción relevante. Ese es el problema de los “unknown unknowns”.

Por tanto, la circularidad se reduce:

```text
antes:
seleccionar pesos correctos antes de comprender.

ahora:
pensar primero y después formular una consulta.
```

Pero no desaparece por completo:

```text
para formular la consulta correcta
todavía necesitas suficiente comprensión inicial.
```

La recuperación debería ser conservadora, multimodal y posiblemente iterativa:

```text
THINK inicial
→ generar varias consultas
→ READ
→ integrar evidencia
→ THINK
→ verificar contradicciones
→ READ AGAIN si procede
```

## 9. Las referencias anteriores sí existen

Verifiqué las principales:

* Ouro existe y sus experimentos atribuyen la mejora de la recurrencia a manipulación del conocimiento, no a mayor capacidad de almacenamiento factual. ([arXiv][4])
* RecurrentGPT existe y muestra precisamente que la recurrencia ingenua degrada; utiliza prelude, coda, proyección recurrente y puertas para diferenciar las vueltas. ([arXiv][5])
* El trabajo de inferencia cache-resident también existe, pero se basa en LLC de escala gigabyte, no en L2 convencional. Confirma que, una vez residentes los pesos, sincronización y separación de KV/atención se vuelven problemas visibles. ([arXiv][6])
* Perceiver y Mamba-2 respaldan las dos adyacencias principales: workspace latente y matrixización de estados, respectivamente. ([arXiv][1])

## 10. El diseño que queda ya no es un MoE

La arquitectura resultante sería más parecida a un pequeño computador neuronal:

$$
S_t\in\mathbb{R}^{m\times d}
$$

### READ

$$
Q_t=\operatorname{Query}(S_{t-1},x_t)
$$

$$
R_t=\operatorname{Retrieve}(M,Q_t)
$$

$$
S_t^{(0)}
=
\operatorname{WriteEvidence}
(S_{t-1},x_t,R_t)
$$

### THINK

$$
S_t^{(r+1)}
=
S_t^{(r)}
+
G_r
\odot
F_\theta
\left(
S_t^{(r)},E_{\text{roles}},E_r
\right)
$$

para:

$$
r=0,\ldots,K_t-1
$$

Los pesos grandes de \(F_\theta\) permanecen shardeados en las L2 privadas. Las \(m\) filas de \(S\) convierten las proyecciones en small-GEMM.

### CHECK

Un slot de control decide:

$$
\text{emitir},\quad
\text{pensar otra vuelta},\quad
\text{consultar otra vez}
$$

### EMIT

$$
p(x_{t+1})
=
\operatorname{LMHead}
\left(
S_{t,\text{output}}
\right)
$$

El banco de RAM debería almacenar principalmente **datos y representaciones**, no millones de fragmentos de pesos neuronales. Así, una omisión de recuperación no corrompe silenciosamente cada transformación del núcleo; se convierte en una ausencia de evidencia, que es más fácil de verificar y corregir.

## 11. El gate de hardware definitivo

Haría dos barridos, no uno.

### Barrido 1: \(K\) fijo

```text
K ∈ {4, 8}
m ∈ {1, 2, 4, 8, 16, 32}
```

Aquí aumentan los FLOPs. Sirve para medir puramente:

* utilización de MAC;
* amortización de dequant;
* transición GEMV→GEMM;
* coste de slots;
* saturación de registros y caché.

### Barrido 2: FLOPs constantes

```text
Km ∈ {32, 64, 96}
```

Ejemplos para \(Km=32\):

```text
(K,m) = (32,1), (8,4), (4,8), (2,16), (1,32)
```

Este barrido mide el intercambio profundidad↔workspace.

En cada punto:

```text
A: pesos compartidos residentes
B: pesos distintos
C: pesos compartidos, expulsión artificial
D: Q4 fusionado
E: pesos predesempaquetados a Q8/int8
```

La variante E permite separar:

```text
ganancia por GEMM
vs.
ganancia por amortizar dequantización
```

Las predicciones falsables correctas serían:

1. La fracción del pico de MAC debe subir claramente entre \(m=1\) y \(m=8\) o \(16\).
2. Las instrucciones de desempaquetado por MAC deben caer al aumentar \(m\).
3. La diferencia A↔B debe crecer principalmente con \(K\), no con \(m\).
4. A↔C debe desaparecer cuando se fuerza la expulsión.
5. A FLOPs constantes, la latencia debería caer al principio y después aplanarse o subir por interacción entre slots.

No afirmaría de antemano una mejora “superlineal”. La predicción más defendible es una curva con rodilla:

```text
m pequeño:
limitado por L2, dequant y baja ocupación.

m intermedio:
small-GEMM eficiente.

m grande:
limitado por cómputo, registros y atención entre slots.
```

# Veredicto

La propuesta de Claude ya no es una corrección menor. Introduce el segundo multiplicador necesario:

$$
\boxed{
\text{reutilización en profundidad }K
\times
\text{reutilización entre slots }m
}
$$

La primera evita volver a DRAM. La segunda evita que el núcleo residente siga atrapado en GEMV.

La tesis corregida sería:

> **Una arquitectura CPU-native debe utilizar un workspace latente matricial, procesado por un núcleo pequeño de pesos compartidos y shardeado entre las L2 privadas. La profundidad \(K\) aporta composición secuencial; los slots \(m\) aportan espacio de trabajo e intensidad aritmética; la RAM aporta capacidad de conocimiento mediante lecturas externas e infrecuentes.**

Lo abierto ya no es principalmente el hardware. Lo verdaderamente difícil es demostrar que:

$$
\boxed{
\text{núcleo procedural pequeño}
+
\text{workspace matricial}
+
\text{memoria externa}
}
$$

puede reproducir el conocimiento distribuido, la fluidez y la generalización que hoy emergen dentro de miles de millones de pesos paramétricos.

[1]: https://arxiv.org/html/2103.03206v2 "Perceiver: General Perception with Iterative Attention"
[2]: https://tridao.me/blog/2024/mamba2-part3-algorithm/ "State Space Duality (Mamba-2) Part III - The Algorithm | Tri Dao"
[3]: https://arxiv.org/abs/2511.21702 "[2511.21702] CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference"
[4]: https://arxiv.org/abs/2510.25741 "[2510.25741] Scaling Latent Reasoning via Looped Language Models"
[5]: https://arxiv.org/html/2608.15062v2 "RecurrentGPT: Expressive Depth through Recurrent Modulation in Transformers"
[6]: https://arxiv.org/abs/2606.25353 "[2606.25353] Cache-Resident LLM Inference in GB-Scale Last-Level Caches"

### GPT 5.6 Sol

Sí. Con el hardware real, hay que fijar tres decisiones antes de interpretar cualquier resultado:

1. **La laptop será plataforma de validación de inferencia, no de entrenamiento principal.**
2. Cuando se diga “modelo looped de \(X\) parámetros”, \(X\) significará **parámetros entrenables únicos del núcleo**, salvo que se indique explícitamente otra cosa. El banco externo se contabilizará aparte.
3. El primer núcleo residente no será de 12 MB ni 24M de pesos: en este chip, el candidato realista está alrededor de **2–2.5 MiB empaquetados en total**, aproximadamente 4M parámetros Q4.

Además, el Ryzen AI 5 330 no contiene cuatro núcleos equivalentes: tiene **1 Zen 5 y 3 Zen 5c**, 4 núcleos/8 hilos, 4 MB de L2 total y 8 MB de L3. El Zen 5 alcanza hasta 4.5 GHz y los Zen 5c hasta 3.4 GHz. El sharding debe balancearse por rendimiento medido, no dividirse automáticamente en cuatro partes iguales. ([AMD][1])

# 1. Redimensionamiento al Ryzen AI 5 330

## La relación L2/DRAM no se reduce linealmente por tener la mitad de núcleos

La relación relevante es:

$$
r_4=
\frac{
BW_{\text{L2,Zen5}}
+
3BW_{\text{L2,Zen5c}}
}{
BW_{\text{DRAM con 4 núcleos}}
}
$$

No es simplemente:

$$
r_4=\frac{4}{8}r_8
$$

porque el denominador también depende del número de núcleos:

$$
BW_{\text{DRAM}}(n)
=
\min
\left(
BW_{\text{canales}},
n\cdot BW_{\text{generado por núcleo}}
\right)
$$

Hay dos casos:

* Si cuatro núcleos ya saturan la memoria, reducir de ocho a cuatro aproximadamente reduce solo el numerador y \(r\) puede caer cerca de la mitad.
* Si cuatro núcleos no saturan la memoria, también cae el denominador y la relación puede mantenerse relativamente alta.

En este procesador aparece además la asimetría Zen 5/Zen 5c y el límite térmico y energético de una laptop. Por tanto:

> **Una relación efectiva de 10–20× sigue siendo plausible, pero no puede utilizarse como supuesto previo. Es una de las primeras mediciones del gate.**

La configuración de memoria puede cambiarla drásticamente. ASUS especifica 16 GB DDR5 integrados y señala que el funcionamiento dual-channel requiere instalar al menos un módulo SO-DIMM. Si la laptop conserva únicamente los 16 GB integrados, la medición de DRAM puede ser sustancialmente distinta de una configuración con el SO-DIMM poblado. Esto debe registrarse junto al resultado. ([ASUS Global][2])

## Presupuesto físico de L2

Aunque existe 1 MiB nominal por núcleo físico, no debemos asignar 1 MiB completo a pesos. También compiten por L2:

* escalas y metadatos Q4;
* activaciones;
* buffers de salida;
* estados intermedios de SwiGLU;
* stack y código;
* contaminación del sistema;
* datos compartidos entre workers.

El primer objetivo razonable es:

$$
0.50\text{–}0.70\text{ MiB de pesos por núcleo}
$$

o:

$$
2.0\text{–}2.8\text{ MiB de núcleo total}
$$

El límite de 0.70 MiB no se considera una verdad arquitectónica. Se barrerá hasta encontrar la rodilla real.

## Un bloque recurrente concreto para esta máquina

Supongamos un bloque con:

* Q, K, V y proyección de salida: \(4d^2\);
* SwiGLU con expansión \(4d\): \(3d(4d)\).

Entonces:

$$
P_{\text{bloque}}
=
4d^2+12d^2
=
16d^2
$$

Asumiendo Q4 y una escala FP16 por cada 32 pesos:

$$
q_{\text{efectivo}}
\approx
0.5+\frac{2}{32}
=
0.5625\text{ bytes/peso}
$$

El barrido queda:

| \(d\) | Parámetros únicos | Q4 + escalas | Promedio por núcleo | Régimen esperado           |
| ----: | ----------------: | -----------: | ------------------: | -------------------------- |
|   384 |             2.36M |     1.27 MiB |            0.32 MiB | L2 holgada                 |
|   512 |             4.19M |     2.25 MiB |            0.56 MiB | **candidato principal L2** |
|   640 |             6.55M |     3.52 MiB |            0.88 MiB | frontera L2                |
|   768 |             9.44M |     5.06 MiB |            1.27 MiB | L3 compartida              |
|  1024 |            16.78M |     9.00 MiB |            2.25 MiB | excede L3; DRAM            |

La tabla no incluye todo el padding de un formato Q4 real, por lo que el ejecutable debe reportar los bytes físicos asignados, no calcularlos solo desde \(P/2\).

La configuración de **\(d=512\)** es especialmente útil:

* aproximadamente 2.25 MiB de pesos totales;
* aproximadamente 0.56 MiB por núcleo con sharding uniforme;
* deja espacio para activaciones y escalas;
* permite comprobar si las cuatro L2 privadas pueden operar como residencia agregada.

La de \(d=640\) debe mostrar dónde empieza a romperse la residencia. \(d=768\) prueba el régimen L3 y \(d=1024\) el régimen DRAM.

## Sharding para 1 Zen 5 + 3 Zen 5c

Primero se mide cada núcleo físico de manera individual con el mismo microkernel. Para cada núcleo:

$$
v_i=\text{MAC/s sostenidos del núcleo }i
$$

Después se asignan filas proporcionalmente:

$$
P_i
=
P_{\text{total}}
\frac{v_i}{\sum_j v_j}
$$

sujeto a:

$$
B_i \le B_{\text{L2 útil},i}
$$

No debe asumirse qué número de CPU lógica corresponde al Zen 5. El harness debe identificar las parejas SMT y medir cada núcleo físico. El worker rápido probablemente recibirá más filas, pero la proporción debe salir del benchmark.

La primera ejecución usa:

```text
4 workers
1 hilo por núcleo físico
afinidad fija
```

Los ocho hilos SMT se prueban después como variante. SMT puede ocultar algunas latencias, pero:

* no añade L2;
* no añade unidades vectoriales;
* comparte el frontend y backend del núcleo;
* puede mejorar o empeorar el small-GEMM.

Por tanto, `8 threads` es una fila experimental, no la configuración base.

# 2. Barrido \(m,K\) adaptado a esta laptop

Conviene separar dos gates.

## H0: microkernel Q4 lineal

Este gate comprueba exclusivamente:

* residencia;
* transición GEMV → small-GEMM;
* amortización del desempaquetado;
* comportamiento L2/L3/DRAM.

Barridos:

```text
m ∈ {1, 2, 4, 8, 16, 32}
K ∈ {1, 2, 4, 8, 16}
bytes de pesos por núcleo:
{256, 384, 512, 640, 768, 896, 1024, 1280} KiB
```

El conjunto \(m=32\) es de estrés. Puede aumentar demasiado la presión de activaciones y registros, pero necesitamos observar dónde se aplana la curva.

## H1: bloque recurrente realista

Usa las dimensiones anteriores:

```text
d ∈ {384, 512, 640, 768, 1024}
m ∈ {1, 4, 8, 16, 32}
K ∈ {1, 2, 4, 6, 8}
```

El subconjunto más probable para una primera arquitectura entrenable es:

```text
d = 512
m ∈ {4, 8, 16}
K ∈ {4, 6, 8}
```

La interacción slot-a-slot cuesta aproximadamente:

$$
O(Km^2d)
$$

mientras que las proyecciones cuestan aproximadamente:

$$
O(KmP)
$$

Con \(m\le16\), \(d=512\) y \(P\approx4.2M\), la atención entre slots es pequeña frente a las proyecciones del bloque.

## Barrido a FLOPs constantes

Para separar profundidad y anchura del workspace:

$$
Km=C
$$

Con \(C=32\):

```text
(K,m):
(32,1)
(16,2)
(8,4)
(4,8)
(2,16)
(1,32)
```

Con \(C=64\):

```text
(16,4)
(8,8)
(4,16)
(2,32)
```

El primer barrido es principalmente físico. No implica que \(K=32\) o \(K=1\) sean buenas configuraciones de aprendizaje.

# 3. Variantes A/B/C exactas

Cada variante ejecuta exactamente las mismas formas, MACs, normalizaciones, no linealidades y barreras.

### A — recurrente residente

```text
K aplicaciones
misma dirección de pesos
mismo núcleo Q4
```

### B — profundidad no compartida

```text
K aplicaciones
K copias distintas de los pesos
mismas formas
mismos FLOPs
```

Para \(d=512\):

$$
K=8
\Rightarrow
18\text{ MiB aproximadamente}
$$

Eso supera ampliamente los 8 MB de L3 y obliga a que el camino no compartido toque DRAM.

### C — recurrente expulsado

Es igual a A, pero se expulsa el núcleo entre aplicaciones.

C es un **control causal**, no un baseline de velocidad. La expulsión debe ocurrir fuera de la región de tiempo atribuida al kernel individual, o sus costes deben registrarse por separado. Puede utilizarse un buffer particionado de 32–64 MiB para destruir la residencia de L2 y L3.

La evidencia deseada es:

```text
A: núcleo residente
B: pesos distintos
C: mismos pesos, residencia destruida
```

y:

$$
A \ll B\approx C
$$

en coste marginal por iteración, especialmente con \(m=1\) o \(m=4\), donde el cómputo todavía no oculta la diferencia de memoria.

# 4. Criterios de paso del gate

No usaría un speedup total arbitrario como única condición. Separaría las propiedades.

## Residencia

Definimos:

$$
c_A(K)=\frac{T_A(K)-T_A(1)}{K-1}
$$

$$
c_B(K)=\frac{T_B(K)-T_B(1)}{K-1}
$$

y:

$$
\rho_{\text{resident}}
=
\frac{c_A(K)}{c_B(K)}
$$

Para \(d=512\), \(m=1\) o \(4\), \(K=8\):

```text
ρ ≤ 0.50  → gate mínimo aprobado
ρ ≤ 0.25  → evidencia fuerte
ρ ≈ 1.00  → no existe ventaja útil de residencia
```

C debe acercarse a B. Si A y C resultan prácticamente iguales, la hipótesis L2-residente ha fallado.

## Matrixización

En la variante A, con el núcleo residente, se mide:

$$
\eta(m)
=
\frac{\text{MAC/s efectivos}}
{\text{pico MAC/s medido}}
$$

El denominador no será el TOPS publicitario. Se obtiene con un kernel pequeño residente en L1/L2 y la misma ruta de instrucciones.

Gate mínimo:

```text
MAC/s(m=8 u 16) / MAC/s(m=1) ≥ 1.5
```

Señal fuerte:

```text
≥ 2.0
```

Si la curva permanece plana desde \(m=1\), la hipótesis de convertir el batch interno en small-GEMM no aporta lo esperado.

## Rodillas de caché

Debe aparecer una transición visible:

```text
d=384/512 → régimen L2
d=640     → frontera
d=768     → L3
d=1024    → DRAM
```

Si no aparece ninguna discontinuidad o degradación progresiva al cruzar esos tamaños, significa que:

* el núcleo no se mantiene residente;
* el sharding no está funcionando;
* las activaciones están expulsando los pesos;
* el kernel no está usando la jerarquía como se esperaba.

## SMT

El resultado de ocho hilos se reporta aparte:

```text
4T físicos
8T SMT
```

SMT puede aceptarse para producción si mejora el tiempo total, pero nunca se contará como duplicación de L2 o de capacidad residente.

# 5. Medición en Windows 11

La prueba debe ser **nativa de Windows**, no ejecutarse primero dentro de WSL2. WSL2 utiliza un kernel Linux dentro de una máquina virtual administrada; eso introduce una capa adicional precisamente donde queremos medir afinidad y contadores de hardware. Puede utilizarse como contraste, no como resultado definitivo. ([Microsoft Learn][3])

AMD uProf funciona en Windows y ofrece CPU profiling, análisis de sistema, IPC y ancho de banda de memoria. La documentación también enumera métricas de L2, L3 y tráfico DRAM para arquitecturas Zen recientes, aunque la disponibilidad exacta debe comprobarse en este Ryzen cliente ejecutando `AMDuProfPcm -h`; parte de la tabla oficial está documentada específicamente para EPYC. ([AMD][4])

El camino de medición sería:

```text
Timing:
QueryPerformanceCounter
+ afinidad fija
+ mediana de múltiples ejecuciones

Ciclos:
RDTSCP por worker, cuando sea útil

Memoria:
AMDuProfPcm.exe -m memory -a ...

Caches:
uProf CPU profile con L2/L3 si el chip/driver los expone
```

AMD documenta en Windows el comando para registrar ancho de banda agregado de memoria con `AMDuProfPcm`. ([AMD Documentation][5])

Incluso si los eventos finos de L2 no estuvieran disponibles, A/B/C sigue siendo falsable mediante:

* tiempo de pared;
* tráfico DRAM agregado;
* tamaño de working set;
* expulsión controlada;
* curva de latencia contra bytes residentes.

El procesador soporta AVX-512 oficialmente, pero el harness debe hacer detección de CPUID y comparar rutas AVX2 y AVX-512 en lugar de asumir que la ruta más ancha será siempre la más rápida bajo el presupuesto térmico de la laptop. ([AMD][1])

# 6. Qué significan exactamente “\(X\) parámetros”

Aquí había una ambigüedad real.

Cuando antes se dijo:

```text
núcleo looped de 24M × 30 vueltas
contra
denso de 720M
```

los **24M eran solo pesos únicos del núcleo recurrente**. El banco externo no estaba incluido. Era una comparación ilustrativa iso-FLOPs, no una comparación justa de capacidad ni almacenamiento.

Por tanto, esa forma de expresarlo debe abandonarse.

El ledger correcto es:

$$
P_{\text{core}}
=
\text{pesos únicos del núcleo recurrente}
$$

$$
P_{\text{shell}}
=
\text{embeddings + prelude + coda + head + gates}
$$

$$
P_{\text{retriever}}
=
\text{parámetros del generador de consultas y recuperación}
$$

$$
P_{\text{memory-learned}}
=
\text{vectores del banco si son entrenables}
$$

$$
B_{\text{memory-static}}
=
\text{hechos, documentos o vectores congelados}
$$

$$
B_{\text{index}}
=
\text{índice ANN, centroides, códigos PQ y metadatos}
$$

Entonces:

$$
P_{\text{trainable-total}}
=
P_{\text{core}}
+
P_{\text{shell}}
+
P_{\text{retriever}}
+
P_{\text{memory-learned}}
$$

y:

$$
B_{\text{persistent-total}}
=
B_{\text{pesos cuantizados}}
+
B_{\text{memory-static}}
+
B_{\text{index}}
$$

Si el banco contiene datos estáticos, no se llama “parámetros”, pero **sí cuenta como almacenamiento del sistema**. Si contiene vectores aprendidos, se reporta simultáneamente como parámetros aprendidos y como bytes persistentes.

## Comparación factorial obligatoria

Para separar el efecto de la recursión del efecto de añadir recuperación:

| Arquitectura | Sin banco | Con el mismo banco |
| ------------ | --------- | ------------------ |
| Densa        | D0        | D1                 |
| Recurrente   | R0        | R1                 |

Esto permite responder preguntas distintas:

$$
R0-D0
\Rightarrow
\text{efecto del núcleo recurrente}
$$

$$
D1-D0
\Rightarrow
\text{efecto de la memoria sobre el denso}
$$

$$
R1-R0
\Rightarrow
\text{efecto de la memoria sobre el recurrente}
$$

$$
(R1-R0)-(D1-D0)
\Rightarrow
\text{interacción real entre recurrencia y banco}
$$

Sin D1, cualquier ganancia de R1 podría ser simplemente RAG convencional añadido al sistema.

## Cuatro regímenes de comparación

No existe una única noción de “igual tamaño”. Deben reportarse cuatro:

### Igual cantidad de parámetros únicos

Compara capacidad paramétrica del núcleo.

### Igual FLOPs por token

Compara eficiencia computacional, aunque el denso tenga más pesos únicos.

### Igual almacenamiento persistente total

Incluye pesos, banco e índice. Es la comparación correcta para afirmar:

> “Este sistema obtiene más calidad con la misma cantidad total de información almacenada”.

### Igual latencia o bytes DRAM por token

Es la comparación de despliegue:

> “Con el mismo presupuesto de hardware, ¿qué sistema produce mejor calidad?”

Por tanto, sí: un núcleo de 4M con un banco de 1 GB puede saber objetivamente más hechos que un denso pequeño. Pero no puede describirse como “modelo de 4M contra modelo de 4M”. Debe describirse como:

```text
4M de núcleo
+ X MB de shell
+ 1 GB de memoria
+ Y MB de índice
```

Eso no invalida la arquitectura. Cambia la afirmación científica:

> No buscamos almacenar más conocimiento por parámetro del núcleo; buscamos obtener más conocimiento útil por byte leído y por unidad de cómputo activo.

# 7. Estrategia de entrenamiento: no desde cero en esta laptop

La decisión debe ser inequívoca:

> **No se entrenará desde cero un modelo lingüístico significativo en el Ryzen AI 5 330.**

La laptop hará:

* H0 y H1;
* profiling;
* ejecución Q4;
* pruebas de caché;
* validación de checkpoints;
* tareas sintéticas diminutas.

El entrenamiento lingüístico se hará en GPU mediante inicialización parcial y destilación desde un maestro denso.

## Corrección sobre el coste de pesos compartidos

El coste depende del baseline.

Frente a aplicar el núcleo una sola vez:

$$
C_{\text{train}}
\propto K
$$

Frente a un Transformer de \(K\) capas no compartidas con las mismas dimensiones:

* los FLOPs forward/backward son aproximadamente comparables;
* la red recurrente tiene muchos menos parámetros únicos;
* tiene menos estados del optimizador;
* pero conserva activaciones de las \(K\) vueltas;
* y sus gradientes atraviesan repetidamente la misma transformación.

Por tanto, weight sharing no hace mágicamente el entrenamiento más barato en FLOPs. Reduce almacenamiento paramétrico, pero no elimina el coste del desenrollado. Universal Transformer y Perceiver demuestran que la recurrencia en profundidad y los latentes compartidos son entrenables, aunque no demuestran nuestra combinación con banco externo y residencia L2. ([arXiv][6])

Además, aquí aparecen **dos recurrencias**:

1. profundidad \(K\) dentro del mismo token;
2. persistencia del workspace entre tokens.

Entrenar ambas mediante BPTT completo sobre conversaciones largas sería costoso e inestable. La estrategia inicial debe evitarlo.

# 8. Programa de entrenamiento propuesto

## T0 — gate físico

No requiere entrenamiento.

```text
Q4 sintético
bloque A/B/C
d=512 como candidato principal
4 threads físicos
m/K barridos
```

Si H0/H1 no pasan, la línea muere antes de gastar GPU.

## T1 — aprendizaje sintético pequeño

Se entrena desde cero únicamente un modelo diminuto:

```text
d = 64–128
m = 4–8
K = 2–4
```

Tareas:

* copiar;
* recuperar una clave;
* composición de dos hechos;
* actualizar slots;
* decidir READ/THINK;
* evitar colapso de slots.

Esto comprueba entrenabilidad, no capacidad lingüística.

## T2 — destilación del núcleo sin banco

Se utiliza un Transformer denso preentrenado como maestro congelado.

El estudiante recibe:

* token actual;
* estado previo;
* slots;
* representaciones resumidas del contexto.

La pérdida puede combinar:

$$
\mathcal L
=
\lambda_{\text{CE}}\mathcal L_{\text{token}}
+
\lambda_{\text{KL}}
D_{\text{KL}}(p_T\|p_S)
+
\lambda_h
\sum_r
\left\|
A_rS^{(r)}-H_T^{(\ell_r)}
\right\|^2
$$

donde \(A_r\) proyecta los slots al espacio de estados del maestro.

El entrenamiento usa currículo:

```text
K = 2
→ K = 4
→ K = 6

m = 4, 8 y 16 como ejecuciones separadas
```

No conviene cambiar simultáneamente \(m\) y \(K\) dentro de una misma corrida inicial.

Para evitar BPTT conversacional completo:

1. se generan trazas del maestro offline;
2. se entrena el estado siguiente usando el estado objetivo anterior;
3. luego se introduce scheduled sampling;
4. finalmente se realizan unrolls cortos de 8, 16 y 32 tokens con truncated BPTT.

La destilación reduce el problema de descubrimiento, pero no elimina los FLOPs de entrenamiento del estudiante.

## T3 — añadir la memoria externa

Primera versión:

```text
banco congelado
keys/values creados offline desde corpus o estados del maestro
retriever pequeño entrenable
núcleo inicialmente congelado
```

Se entrena primero:

* formulación de consultas;
* selección top-\(k\);
* escritura de evidencia en slots;
* detección de recuperación insuficiente.

Después se descongela parcialmente el núcleo y se afina de forma conjunta.

El banco debe construirse exclusivamente desde datos permitidos para entrenamiento. Su cobertura y tamaño se reportan, evitando que una prueba de “conocimiento” sea simplemente recuperación de respuestas de evaluación almacenadas.

## T4 — cuantización

El orden adecuado es:

```text
BF16/FP16 estable
→ Q8
→ Q4 con fine-tuning consciente de cuantización
→ validación nativa en laptop
```

No comenzaría entrenando directamente con el núcleo Q4. La recurrencia puede amplificar errores repetidos, y primero necesitamos separar fallos de optimización de fallos de cuantización.

# 9. Qué valida realmente esta laptop

El Ryzen AI 5 330 puede responder, con bastante limpieza, estas preguntas:

1. ¿Un bloque de aproximadamente 2.25 MiB puede permanecer distribuido en las cuatro L2?
2. ¿La repetición compartida evita tráfico DRAM frente a capas distintas?
3. ¿\(m=8\) o \(m=16\) convierte suficientemente el trabajo en small-GEMM?
4. ¿Dónde están las rodillas L2 y L3?
5. ¿Cuánto perjudica la heterogeneidad 1×Zen 5 + 3×Zen 5c?
6. ¿SMT ayuda o estorba?
7. ¿Qué \(d,m,K\) ofrece el mejor tiempo real bajo 15–28 W?

No puede demostrar:

* escalado lineal a 16 o 32 núcleos;
* calidad lingüística;
* capacidad del banco;
* estabilidad de entrenamiento;
* equivalencia con un Transformer grande.

Pero es suficiente para matar o conservar **el mecanismo físico central**.

# Decisión final

El contrato queda así:

```text
Plataforma de gate:
Ryzen AI 5 330 bare metal, Windows 11

Workers primarios:
4, uno por núcleo físico

Núcleo inicial:
d=512
≈4.19M pesos únicos
≈2.25 MiB Q4 + escalas
≈0.56 MiB promedio por núcleo

Workspace:
m ∈ {1, 4, 8, 16, 32}

Recurrencia:
K ∈ {1, 2, 4, 6, 8, 16}

Comparaciones:
A residente
B untied
C expulsado
4T vs 8T

Entrenamiento:
ninguno para el gate
destilación en GPU para la fase lingüística

Contabilidad:
núcleo, shell, retriever, banco e índice separados
```

Y la afirmación que se intentará probar no es:

> “Un modelo de 4M equivale a uno de cientos de millones”.

Es:

> **Un núcleo procedural de aproximadamente 4M parámetros puede reutilizarse desde caché con suficiente eficiencia para dejar la mayor parte del presupuesto de DRAM disponible para memoria externa; después habrá que demostrar por separado que esa memoria puede sustituir una fracción significativa del conocimiento paramétrico.**

Con esas definiciones, sí vale la pena ejecutar el primer gate en esa laptop.

[1]: https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-5-330.html?utm_source=chatgpt.com "AMD Ryzen™ AI 5 330"
[2]: https://www.asus.com/us/laptops/for-home/vivobook/asus-vivobook-16-m1607/techspec/?utm_source=chatgpt.com "ASUS Vivobook 16 (M1607) - Tech Specs"
[3]: https://learn.microsoft.com/en-us/windows/wsl/compare-versions?utm_source=chatgpt.com "Comparing WSL Versions"
[4]: https://www.amd.com/en/developer/uprof.html?utm_source=chatgpt.com "AMD μProf"
[5]: https://docs.amd.com/r/en-US/57368-uProf-user-guide/4.5.2.2.-Windows-Commands?utm_source=chatgpt.com "4.5.2.2. Windows Commands"
[6]: https://arxiv.org/abs/1807.03819?utm_source=chatgpt.com "Universal Transformers"
