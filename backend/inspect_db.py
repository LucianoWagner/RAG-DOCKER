"""
Script de inspección de la BD Vectorial.
Extrae 3 embeddings por categoría directamente de ChromaDB.
"""

from app.retrieval.vector_store import get_chroma_client
from app.config import get_settings

def inspect_chromadb():
    settings = get_settings()
    client = get_chroma_client()
    
    print(f"Conectando a {settings.chroma_host}:{settings.chroma_port}...")
    try:
        col = client.get_collection(settings.chroma_collection_name)
    except Exception as e:
        print(f"Error conectando a la colección: {e}")
        return

    print(f"\n📊 Total de chunks guardados en la BD: {col.count()}")
    print("=" * 80)

    categorias = [
        "cli_reference", 
        "configuration", 
        "getting_started", 
        "installation", 
        "troubleshooting"
    ]

    for cat in categorias:
        print(f"\n\033[93m>>> RECUPERANDO 3 CHUNKS DE: {cat.upper()} \033[0m")
        print("-" * 80)
        
        # Le pedimos nativamente a ChromaDB que nos filtre por esa categoría
        resultados = col.get(
            where={"category": cat},
            limit=3,
            include=['metadatas', 'documents'] # Podríamos sumar 'embeddings' pero llena la consola
        )

        # Si no encontró nada en esa categoría
        if not resultados["ids"]:
            print("   No se encontraron chunks en esta categoría.")
            continue

        # Imprimir lo encontrado
        for i in range(len(resultados["ids"])):
            chunk_id = resultados["ids"][i]
            texto = resultados["documents"][i]
            meta = resultados["metadatas"][i]
            
            print(f"📦 ID: {chunk_id}")
            print(f"   🏷️ Meta: {meta['platform'].upper()} | Archivo: {meta['source_file']}")
            print(f"   📝 Título Extraído: '{meta.get('doc_title', '')}'")
            # Mostrar solo los primeros 120 caracteres para no ensuciar tanto la pantalla
            texto_preview = texto[:120].replace('\n', ' ') + "..."
            print(f"   💬 Contenido: {texto_preview}\n")


if __name__ == "__main__":
    inspect_chromadb()
