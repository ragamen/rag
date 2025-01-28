from supabase import create_client, Client
import numpy as np
import streamlit as st
from datetime import datetime

class SupabaseManager:
    def __init__(self):
        self.client: Client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    
    def insert_document(self, fuente_id: int, content_hash: str, contenido: str, embedding: list, metadata: dict):
        """Inserta en tabla documentos con referencia a fuente"""
        try:
            return self.client.table('documentos').insert({
                "fuente_id": fuente_id,
                "content_hash": content_hash,
                "contenido": contenido,
                "embedding": embedding,
                "metadata": metadata
            }).execute()
        except Exception as e:
            st.error(f"Error insertando documento: {str(e)}")
            return None
    def get_document_by_id(self, document_id: int):
        """
        Busca un documento por su ID y devuelve el fuente_id asociado.
        
        :param document_id: El ID del documento a buscar.
        :return: El fuente_id del documento, o None si no se encuentra.
        """
        try:
            # Realizar la consulta a la tabla documentos
            response = self.client.table('documentos').select('fuente_id').eq('id', document_id).execute()
            
            # Verificar si se encontró el documento
            if response.data:
                return response.data[0]['fuente_id']  # Devolver el fuente_id
            else:
                st.warning(f"No se encontró un documento con ID {document_id}.")
                return None
                
        except Exception as e:
            st.error(f"Error buscando documento por ID: {str(e)}")
            return None   
            
    def search_documents(self, query_embedding: np.ndarray, top_k: int = 5):
        try:
            response = self.client.rpc(
                'search_documents',
                {
                    'query_embedding': query_embedding.tolist(),
                    'top_k': top_k
                }
            ).execute()
            
            # Normalizar estructura
            normalized = []
            for doc in response.data:
                normalized.append({
                "id": doc["id"],
                "contenido": doc["contenido"],
                "metadata": doc.get("metadata", {}),
                "similarity": doc.get("similarity", 0.0),
                "fuente_id": doc.get("fuente_id")  # Incluir fuente_id
            })
                # Asegurar campos mínimos
                normalized[-1]["metadata"].setdefault("exact_pages", [])
                normalized[-1]["metadata"].setdefault("paginas_total", 1)
                
            return normalized
            
        except Exception as e:
            error_msg = f"""
            🔴 Error de búsqueda vectorial:
            - Tipo embedding: {type(query_embedding)}
            - Forma embedding: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'N/A'}
            - Error: {str(e)}
            """
            st.error(error_msg)
            return []
        
    def is_document_processed(self, content_hash: str) -> bool:
        """Verifica si un documento ya existe en fuentes"""
        res = self.client.table('fuentes').select('id').eq('content_hash', content_hash).execute()
        return len(res.data) > 0

    def insert_fuente(self, metadata: dict, content_hash: str) -> int:
        """Inserta en tabla fuentes y retorna el ID generado"""
        try:
            data = {
                "content_hash": content_hash,
                "titulo": metadata["title"],
                "autor": metadata["author"],
                "categoria": metadata["categoria"],
                "paginas_total": metadata["paginas_total"],
                "anio_publicacion": metadata["anio_publicacion"]
            }
            response = self.client.table('fuentes').insert(data).execute()
            return response.data[0]['id']
        except Exception as e:
            st.error(f"Error insertando fuente: {str(e)}")
            return -1
    
    def get_fuente(self, fuente_id: int):
        try:
            response = self.client.table('fuentes').select('*').eq('id', fuente_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            st.error(f"Error obteniendo fuente: {str(e)}")
            return None
