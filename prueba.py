import streamlit as st
from supabase import create_client, Client

# Configuración de Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Título de la aplicación
st.title("📚 Traer Campos de Texto de la Tabla 'fuentes'")

# Botón para cargar datos
if st.button("Cargar Datos de la Tabla 'fuentes'"):
    try:
        # Obtener datos de la tabla 'fuentes'
        response = supabase.table('fuentes').select("*").execute()
        
        # Verificar si hay datos
        if response.data:
            st.write("### Datos de la Tabla 'fuentes':")
            
            # Mostrar los datos en una tabla
            st.table(response.data)
        else:
            st.warning("No se encontraron datos en la tabla 'fuentes'.")
    
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")