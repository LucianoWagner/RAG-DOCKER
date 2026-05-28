# Reporte de evaluacion RAGAS - corrida quick Groq 70B

Archivo base:

`backend/tests/ragas_report_llama_3_3_70b_versatile_judge_groq_llama_3_1_8b_instant_20260528_1839.csv`

Dataset:

- Modo: `quick`
- Preguntas: 10
- Generador: `llama-3.3-70b-versatile`
- Juez RAGAS: Groq `llama-3.1-8b-instant`
- Fecha: 2026-05-28 18:39

## Resumen ejecutivo

La corrida muestra un sistema razonablemente grounded, pero todavia irregular en retrieval y en relevancia final.

Promedios generales:

| Metrica | Score | Lectura |
|---|---:|---|
| Faithfulness | 0.723 | Aceptable. El LLM suele apoyarse en los chunks, pero hay alucinacion o citas debiles en varios casos. |
| Answer relevancy | 0.573 | Flojo. Varias respuestas no atacan directamente la pregunta o se abstienen cuando deberian responder. |
| Context precision | 0.653 | Medio. El retriever trae informacion util, pero tambien mucho ruido o chunks mal posicionados. |
| Context recall | 0.878 | Bueno. En general aparece la informacion necesaria, aunque hay fallas puntuales. |
| Latencia promedio | 2.30s | Muy buena para una evaluacion con Groq. |

Mi lectura: el problema principal no es el LLM generador. El problema esta mas arriba: algunos chunks recuperados son pobres, incompletos o ruidosos. Cuando el contexto es bueno, la respuesta sale muy bien.

## Resultados por categoria

| Categoria | Faithfulness | Answer relevancy | Context precision | Context recall | Diagnostico |
|---|---:|---:|---:|---:|---|
| installation | 0.543 | 0.000 | 0.817 | 1.000 | El retrieval encuentra algo relacionado, pero las respuestas no satisfacen la pregunta. Caso Windows es el mas claro. |
| getting_started | 0.833 | 0.924 | 0.944 | 1.000 | Muy buen comportamiento. Retrieval y generacion funcionan bien. |
| cli_reference | 0.675 | 0.486 | 0.292 | 0.833 | Irregular. Hay queries CLI donde falta el comando exacto o queda mal rankeado. |
| troubleshooting | 0.688 | 0.477 | 0.371 | 0.556 | Debil. El sistema trae ruido en errores y troubleshooting. |
| configuration | 0.875 | 0.979 | 0.840 | 1.000 | Categoria fuerte. Buen contexto y buenas respuestas. |

## Resultados por pregunta

| Pregunta | Categoria | Faithfulness | Answer relevancy | Context precision | Context recall | Lectura |
|---|---|---:|---:|---:|---:|---|
| How do I install Docker Desktop on Windows? | installation | 0.286 | 0.000 | 1.000 | 1.000 | Caso enganoso: RAGAS ve contexto relevante, pero el chunk principal parece ser solo un titulo. La respuesta se abstiene y no resuelve. |
| What are the system requirements for Docker Desktop on Mac? | installation | 0.800 | 0.000 | 0.633 | 1.000 | La respuesta parece bastante alineada; el 0.000 en relevancia puede ser ruido del juez, pero conviene revisar. |
| How do I run a container in detached mode with a specific port mapping? | getting_started | 1.000 | 0.974 | 0.970 | 1.000 | Excelente. Este es el comportamiento esperado. |
| What does the `docker compose up` command do? | getting_started | 0.667 | 0.874 | 0.917 | 1.000 | Bueno, aunque mezcla algun detalle no central. |
| How do I list all Docker images on my system? | cli_reference | 0.778 | 0.971 | 0.000 | 1.000 | La respuesta es buena, pero RAGAS penaliza context precision. Puede indicar mal orden o juicio inconsistente. |
| How do I remove all stopped containers? | cli_reference | 0.571 | 0.000 | 0.583 | 0.667 | Falla real. No recupero bien `docker container prune`, entonces responde que no hay comando directo. |
| What should I do if I get `Unable to connect to the Docker daemon`? | troubleshooting | 0.375 | 0.000 | 0.143 | 1.000 | Retrieval ruidoso. Hay contexto parcialmente relevante, pero mal rankeado o mezclado con release notes. |
| Docker Desktop fails to start after installing antivirus software. | troubleshooting | 1.000 | 0.953 | 0.600 | 0.111 | Respuesta buena, pero recall bajo: el contexto no cubre todo lo que el ground truth espera. |
| How do I configure Docker to start on boot on Linux? | configuration | 0.875 | 0.982 | 1.000 | 1.000 | Excelente. |
| Can I use Docker without sudo privileges on Linux? | configuration | 0.875 | 0.975 | 0.679 | 1.000 | Buena respuesta, con algo de ruido contextual. |

## Lo que esta funcionando bien

1. Cuando el contexto recuperado es correcto, el modelo responde bien.

   Ejemplos:

   - Detached mode + port mapping.
   - `docker compose up`.
   - Start on boot con systemd.
   - Docker sin sudo en Linux.

2. La fidelidad general esta por encima de 0.7.

   Esto indica que el prompt restrictivo y la temperatura 0.0 ayudan. El LLM no esta inventando descontroladamente.

3. El recall general es alto.

   En muchas preguntas, la informacion necesaria aparece en algun lugar del top 10.

4. La latencia es buena.

   2.30s promedio para generacion con Groq es muy usable.

## Problemas principales

### 1. Chunks demasiado pobres o incompletos

El caso mas claro es:

`How do I install Docker Desktop on Windows?`

El primer contexto recuperado es:

```text
## Install Docker Desktop on Windows
```

Eso es solo un encabezado. El LLM hizo lo correcto desde su prompt restrictivo: dijo que no tenia detalles suficientes. El problema no fue Groq, fue el contexto.

Impacto:

- Baja `answer_relevancy`.
- Baja `faithfulness`.
- El usuario recibe una abstencion aunque el corpus probablemente tenga la pagina correcta.

Accion recomendada:

- Filtrar chunks demasiado cortos.
- O mejor: fusionar chunks que son solo encabezado con el chunk siguiente del mismo documento.

### 2. Consultas CLI necesitan mejor lexical matching

Caso:

`How do I remove all stopped containers?`

La respuesta esperada contiene:

```text
docker container prune
```

Pero el sistema respondio que la documentacion no detalla un comando directo. Esto indica que el chunk correcto no entro, o entro mal rankeado.

Acciones recomendadas:

- Aumentar peso BM25 para pruebas CLI, por ejemplo de `0.6` a `0.7`.
- Agregar query expansion simple para comandos frecuentes:
  - "remove all stopped containers" -> "docker container prune stopped containers"
  - "list images" -> "docker images docker image ls"
- Revisar si el corpus incluye paginas de referencia CLI completas.

### 3. Troubleshooting trae demasiado ruido

Caso:

`Unable to connect to the Docker daemon`

Tiene:

```text
faithfulness: 0.375
answer_relevancy: 0.000
context_precision: 0.143
```

Esto sugiere que la busqueda trajo documentos relacionados a Docker Desktop, sockets o release notes, pero no una respuesta limpia y ordenada.

Acciones recomendadas:

- Mejorar metadata y filtrado por categoria `troubleshooting`.
- Dar boost a documentos cuyo path o titulo contenga `troubleshoot`, `daemon`, `socket`, `linux-postinstall`.
- Evitar que release notes compitan tan fuerte para preguntas de soporte.

### 4. RAGAS con juez Groq 8B tiene ruido

Algunos scores parecen discutibles:

- Mac system requirements tiene answer relevancy 0.000, aunque la respuesta contesta bastante bien.
- List images tiene context precision 0.000, aunque la respuesta final es relevante.

Esto no invalida la corrida, pero significa que no hay que tomar cada numero como verdad absoluta. Conviene mirar:

- promedio por categoria,
- ejemplos concretos,
- respuestas y contextos recuperados,
- tendencia entre corridas.

## Cambios recomendados

### Prioridad alta

#### A. Evitar chunks header-only

Problema:

Chunks como:

```text
## Install Docker Desktop on Windows
```

no sirven como evidencia por si solos.

Opciones:

1. Filtrar chunks con menos de N caracteres utiles.
2. Fusionar encabezados sueltos con el siguiente chunk del mismo documento.
3. En `chunker.py`, postprocesar chunks despues de `split_documents()`.

Recomendacion concreta:

- Implementar postprocesado en `chunker.py`.
- Si un chunk tiene menos de 80 caracteres y parece solo heading, concatenarlo con el siguiente chunk del mismo `source_file`.

#### B. Guardar metadata de chunks en la fase collect

Hoy el JSON de evaluacion guarda solo textos de contextos. Para diagnosticar retrieval se necesita tambien:

- `source_file`
- `doc_title`
- `section_header`
- `rerank_score`
- `rerank_position`
- `category`
- `platform`

Esto permitiria saber exactamente que archivo fallo, no solo el texto recuperado.

#### C. Agregar tests especificos de retrieval para preguntas criticas

Antes de mirar generacion, validar que el retriever trae el chunk correcto.

Casos minimos:

| Query | Debe recuperar |
|---|---|
| `How do I remove all stopped containers?` | `docker container prune` |
| `How do I install Docker Desktop on Windows?` | pagina real de instalacion Windows con pasos |
| `Unable to connect to the Docker daemon` | socket/daemon/systemctl o doc troubleshooting adecuada |
| `list all Docker images` | `docker images` o `docker image ls` |

### Prioridad media

#### D. Ajustar retrieval hibrido por tipo de query

Hoy los pesos son fijos:

```python
semantic_weight = 0.4
bm25_weight = 0.6
```

Para CLI y errores literales, BM25 deberia pesar mas. Para preguntas conceptuales, semantico puede pesar mas.

Idea:

```text
si query contiene comando/error/flag -> BM25 0.75, semantico 0.25
si query conceptual -> BM25 0.5, semantico 0.5
```

#### E. Reducir ruido de release notes

Las release notes aparecen en varias preguntas donde no deberian dominar.

Opciones:

- bajar peso a documentos cuyo path contenga `release-notes`;
- excluir release notes salvo que la query mencione versiones, changelog, update, release, bug fix;
- usar metadata `doc_type` o `category` para penalizar.

#### F. Mejorar query rewriting

Antes del retrieval, se puede generar una query expandida corta:

```text
"How do I remove all stopped containers?"
-> "docker container prune remove stopped containers"
```

Esto mejora BM25 sin tocar embeddings.

### Prioridad baja

#### G. Verificacion post-generacion de citas

El sistema parsea `[Source N]`, pero no verifica que cada afirmacion este realmente soportada por esa source.

Se podria agregar luego:

```text
claim + cited chunk -> SUPPORTED / UNSUPPORTED
```

No lo pondria primero. Antes conviene arreglar retrieval/chunking.

## Plan recomendado de trabajo

1. Arreglar chunks header-only.
2. Reingestar corpus.
3. Repetir corrida quick con el mismo comando.
4. Comparar especialmente:
   - Windows install,
   - remove stopped containers,
   - daemon error.
5. Agregar metadata al JSON de `collect`.
6. Crear tests unitarios de retrieval para esos casos.
7. Recien despues ajustar pesos BM25/semantico o query expansion.

## Criterios de exito para la proxima corrida

Para considerar que mejoro de forma real:

| Metrica | Objetivo quick |
|---|---:|
| Faithfulness | >= 0.80 |
| Answer relevancy | >= 0.75 |
| Context precision | >= 0.75 |
| Context recall | >= 0.90 |

Ademas, estas preguntas deberian dejar de fallar:

- `How do I install Docker Desktop on Windows?`
- `How do I remove all stopped containers?`
- `What should I do if I get the error 'Unable to connect to the Docker daemon'?`

## Conclusion

La corrida no es mala. El sistema ya muestra una base solida: cuando recupera buen contexto, responde bien y rapido. El cuello de botella actual es retrieval/chunking, no Groq.

El cambio mas rentable es corregir la calidad de los chunks y diagnosticar con metadata completa. Despues tiene sentido tocar pesos de busqueda hibrida o agregar query expansion para comandos.

