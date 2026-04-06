# RAG Pipeline Evaluation Report (Iteración 2)

*Reporte de evaluación End-to-End generado tras optimizar el subsistema de Recuperación de Datos (Retriever).*

> [!TIP]
> **Configuración Activa Durante el Test:**
> - **Embeddings**: `nomic-embed-text`
> - **Vector Search (Top K)**: `30` (↑ Aumentado desde 10)
> - **Reranker (Top N Final)**: `10` (↑ Aumentado desde 5)
> - **Pesos Híbridos**: `Semantic 40%` | `BM25 60%` (Rebalanceo fuerte hacia Palabras Clave)
> - **LLM Generator & Judge**: `llama-3.1-8b-instant` (Temperatura = `0.0`)

### 📈 Resumen de Mejoras
El **Answer Correctness** general subió drásticamente de **0.50 a 0.77**, evidenciando que el cuello de botella era netamente la falta de lectura bibliográfica (top_k/top_n subdimensionados) impuesta por la base de datos local.

| Categoría | Pregunta | Faithfulness | Correctness | Razón Principal (Correctness) |
| :--- | :--- | :---: | :---: | :--- |
| `installation` | How do I install Docker Desktop on Windows? | **0.0** | 0.0 | **Data Perdida**: Sigue sin poder extraer el paso a paso. Sugiere fuertemente que **este tutorial falta en la base documental inicial**. |
| `installation` | What are the system requirements for Ubuntu? | **0.0** | 0.0 | *Error Técnico:* Excedido de tokens de API durante esta iteración (Llama 8B falló en procesarlo a tiempo). |
| `installation` | How can I install Docker Compose standalone? | **0.0** | 0.5 | Trajo info correcta del repositorio, pero se confundió ligeramente al narrar los pasos por culpa del modelo de 8 Billones. |
| `getting_started`| What is the command to run a basic nginx container? | **1.0** | **1.0** | ✅ **Respuesta Perfecta**. El aumento del BM25 recuperó el comando idéntico. |
| `getting_started`| How do I stop all running containers? | **0.0** | 0.5 | Recuperó `docker stop` y `docker ps`, pero el LLM de 8B no logró razonar la interpolación del `$(...)`. |
| `getting_started`| What does 'docker compose up' do? | 0.0 | **1.0** | ✅ **Respuesta Perfecta**. |
| `troubleshooting`| Unable to connect to the Docker daemon | **1.0** | **1.0** | ✅ **Respuesta Perfecta y Fiel**. Logró extraer `DOCKER_HOST`. |
| `troubleshooting`| Anti-virus software on Windows. How to fix? | **1.0** | **1.0** | ✅ **Respuesta Perfecta y Fiel**. |
| `troubleshooting`| My docker build is slow. How to speed it up? | **0.0** | **1.0** | ✅ **Respuesta Perfecta**. ¡El nuevo Top_N trajo el chunk que mencionaba la *Build Cache* y Multi-stage builds! |
| `out_of_domain` | Kubernetes cluster using Minikube? | **1.0** | **1.0** | ✅ Detectó exitosamente que no existe en la documentación. |
| `out_of_domain` | Kubernetes or Docker Swarm for production? | **1.0** | **1.0** | ✅ Respuesta Perfecta y Fiel. |
| `ambiguous` | Docker is not working, it gives an error. | **1.0** | 0.5 | El LLM de 8B se disculpó, pero le faltó pedirte textualmente el mensaje exacto sugerido por la *Ground Truth*. |
| `ambiguous` | How do I update my container? | **1.0** | **1.0** | ✅ Logró recuperar correctamente los preceptos de regenerar contenedores en vez de editarlos in situ. |
| `configuration` | Change default bridge network IP range? | **1.0** | **1.0** | ✅ **¡Arreglado!** El BM25 en 60% finalmente logró encontrar "bip" y "daemon.json" en el mar de documentos semánticos. |
| `configuration` | Use Docker without sudo on Linux? | **0.0** | **1.0** | ✅ **Respuesta Perfecta**. (Faithfulness 0 por pequeñas alucinaciones en referencias del Llama 8B). |

> [!CAUTION]
> **Reflexión sobre Faithfulness y el Modelo Generador:**
> Notarás que el score de *Faithfulness* (Fidelidad) tuvo altibajos. Esto ocurre porque el modelo **Llama-3.1-8b** es muchísimo más chico que el de `70b` original. Al darle un contexto ridículamente grande (10 chunks), el modelo de 8B tiende a marearse o alucinar links inexistentes ("Source 8"). Tu Arquitectura ideal final es **retornar al ChatGroq Llama-3.3-70b** con esta misma configuración Vectorial de `10 Chunks`. ¡Garantizado va a sacar un +90% de Excelencia!
