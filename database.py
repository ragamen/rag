from supabase import create_client
import streamlit as st

class SupabaseManager:
    def __init__(self):
        self.client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    
    def insert_document(self, content_hash: str, contenido: str, embedding: list, metadata: dict):
        return self.client.table('documentos').insert({
            "content_hash": content_hash,
            "contenido": contenido,
            "embedding": embedding,
            "metadata": metadata
        }).execute()

    def search_documents(self, query_embedding: list, top_k: int = 5):
        return self.client.rpc('search_documents', {
            'query_embedding': query_embedding,
            'top_k': top_k
        }).execute()