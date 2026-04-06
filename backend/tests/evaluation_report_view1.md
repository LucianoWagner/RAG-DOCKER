# RAG Pipeline Evaluation Report

*Below is the detailed breakdown of the CSV report formatted as an easy-to-read table. Metrics range from `0.0` (Fail) to `1.0` (Perfect).*

> [!TIP]
> **Configuración Activa Durante el Test:**
> - **Embeddings**: `nomic-embed-text`
> - **Vector Search (Top K)**: `10`
> - **Reranker (Top N Final)**: `5`
> - **Pesos Híbridos**: `Semantic 60%` | `BM25 40%`
> - **LLM Generator**: `llama-3.3-70b-versatile` (Temperatura = `0.0`)
> - **Judge Evaluator**: `llama-3.3-70b-versatile` (Temperatura = `0.0`)

| Categoría | Pregunta | Faithfulness | Correctness | Razón Principal (Correctness) |
| :--- | :--- | :---: | :---: | :--- |
| `installation` | How do I install Docker Desktop on Windows? | **1.0** | 0.0 | El contexto no trajo los comandos ni pasos de instalación para Windows. |
| `installation` | What are the system requirements for Ubuntu? | **1.0** | 0.0 | El recuperador no encontró qué arquitecturas soportaba Ubuntu. |
| `installation` | How can I install Docker Compose standalone? | **1.0** | 0.5 | Falta de documentación rigurosa encontrada por el retriever. |
| `getting_started`| What is the command to run a basic nginx container? | **1.0** | 0.5 | El snippet no coincidió exactamente con el "Ground Truth". |
| `getting_started`| How do I stop all running containers? | **1.0** | 0.0 | No encontró el comando de `docker ps -aq` en los párrafos recuperados. |
| `getting_started`| What does 'docker compose up' do? | 0.0 | **1.0** | *Alucinación*. Inventó redes abstractas; le atinó a la verdad, pero mintiendo al contexto. |
| `troubleshooting`| Unable to connect to the Docker daemon | **1.0** | **1.0** | ✅ Respuesta Perfecta y Fiel. |
| `troubleshooting`| Anti-virus software on Windows. How to fix? | **1.0** | **1.0** | ✅ Respuesta Perfecta y Fiel. |
| `troubleshooting`| My docker build is slow. How to speed it up? | **1.0** | 0.5 | Información devuelta fue parcialmente útil pero omitió `.dockerignore`. |
| `out_of_domain` | Kubernetes cluster using Minikube? | **1.0** | 0.5 | Responde honestamente que no sabe sobre Minikube. |
| `out_of_domain` | Kubernetes or Docker Swarm for production? | **1.0** | **1.0** | ✅ Respuesta Perfecta y Fiel. |
| `ambiguous` | Docker is not working, it gives an error. | **1.0** | 0.5 | El LLM asumió que era de WSL. |
| `ambiguous` | How do I update my container? | **1.0** | 0.5 | Respuesta vaga debido a chunks de otra sección. |
| `configuration` | Change default bridge network IP range? | **1.0** | 0.0 | No encontró mención a `bip` en `daemon.json`. |
| `configuration` | Use Docker without sudo on Linux? | **1.0** | **1.0** | ✅ Respuesta Perfecta y Fiel. |

> [!NOTE]
> **Diagnóstico del Arquiteco:**
> Tu pipeline saca muy fácilmente notas de 1.0 / 1.0 cada vez que el Retriever hace bien su trabajo (ej. Troubleshootings). Toda la penalidad de Correctness (0.0 ó 0.5) sucede cuando el modelo no encuentra instrucciones técnicas en sus Top-5 Chunks para poder basarse fielmente.
