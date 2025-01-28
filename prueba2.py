import streamlit as st
from supabase_manager import SupabaseManager

# Título de la aplicación
st.title("📄 Traer Datos de la Tabla 'documentos'")

# Crear una instancia de SupabaseManager
supabase_manager = SupabaseManager()

# Botón para cargar datos
if st.button("Cargar Datos de la Tabla 'documentos'"):
    try:
        # Obtener datos de la tabla 'documentos'
        response = supabase_manager.client.table('documentos').select("*").execute()
        
        # Verificar si hay datos
        if response.data:
            st.write("### Datos de la Tabla 'documentos':")
            
            # Mostrar los datos en una tabla
            st.table(response.data)
        else:
            st.warning("No se encontraron datos en la tabla 'documentos'.")
    
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")