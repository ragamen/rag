import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase_manager import SupabaseManager
import numpy as np

# Título de la aplicación
st.title("📄 Búsqueda de Documentos por Similitud de Embeddings")

# Inicializar el embedder y SupabaseManager
embedder = SentenceTransformer('all-MiniLM-L6-v2')
supabase_manager = SupabaseManager()

# Campo de texto para ingresar la consulta
query = st.text_input("Ingresa tu consulta:", "lista de enfermedades virales")

# Definir el número de resultados a mostrar (top_k)
top_k = 5

# Botón para realizar la búsqueda
if st.button("Buscar Documentos Similares"):
    try:
        # 1. Generar el embedding de la consulta
        query_embed = embedder.encode([query])[0]
        st.write("✅ Embedding de la consulta generado correctamente.")

        # 2. Búsqueda Vectorial
        vector_results = supabase_manager.search_documents(query_embed, top_k)
        st.write(f"🔍 Resultados vectoriales encontrados: {len(vector_results)}")

        # 3. Búsqueda Léxica
        lexical_response = supabase_manager.client.rpc(
            'search_lexical',
            {
                'query_text': query,
                'top_k': top_k
            }
        ).execute()
        st.write(f"🔍 Resultados léxicos encontrados: {len(lexical_response.data)}")

        # 4. Normalizar resultados léxicos
        lexical_results = []
        for item in lexical_response.data:
            lexical_results.append({
                "id": item["id"],
                "contenido": item["contenido"],
                "metadata": item.get("metadata", {}),
                "similarity": 0.7,  # Valor por defecto para compatibilidad
                "fuente_id": item.get("fuente_id")  # Incluir fuente_id
            })

        # 5. Combinar y ordenar resultados
        combined_results = vector_results + lexical_results
        sorted_results = sorted(
            combined_results,
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )[:top_k]

        # 6. Mostrar los resultados
        if sorted_results:
            st.write("### Resultados de la Búsqueda:")
            for result in sorted_results:
                st.write(f"**ID:** {result['id']}")
                st.write(f"**Contenido:** {result['contenido']}")
                st.write(f"**Similitud:** {result['similarity']:.4f}")
                st.write(f"**Fuente ID:** {result.get('fuente_id', 'N/A')}")  # Mostrar fuente_id
                st.write(f"**Metadatos:** {result['metadata']}")
                st.write("---")
        else:
            st.warning("No se encontraron documentos similares.")

    except Exception as e:
        st.error(f"❌ Error al realizar la búsqueda: {str(e)}")