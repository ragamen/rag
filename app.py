from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Tried to instantiate class '__path__._path'")
import json
import requests
import numpy as np
import streamlit as st
from datetime import datetime
from collections import defaultdict
import hashlib
import re
import fitz  # PyMuPDF
import struct
from docx import Document
# Agrega estas importaciones al inicio del archivo
from docx.table import Table
from docx.document import Document as DocxDocument
from docx.text.paragraph import Paragraph  # Importar Paragraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from supabase_manager import SupabaseManager
#embedder = SentenceTransformer('all-MiniLM-L6-v2')
try:
    # Intenta acceder a la clave secreta
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    #st.write(f"Tu clave API es: {DEEPSEEK_API_KEY}")
except KeyError:
    st.error("La clave API no está configurada. Por favor, verifica el archivo secrets.toml.")

#DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# st.error(f"Clave Api key {DEEPSEEK_API_KEY}")

# Configuración inicial
embedder = SentenceTransformer('all-MiniLM-L6-v2')


class SessionState:
    def __init__(self):
        # Configuración FAISS (In-memory)
        self.faiss_index = faiss.IndexFlatL2(384)
        self.faiss_index = faiss.IndexIDMap(self.faiss_index)
        
        # Conexión Supabase (Persistencia)
        self.supabase = SupabaseManager()
        
        # Metadata y estado
        self.metadata_map = {}
        self.document_store = defaultdict(list)
        self.chat_history = []
        self.uploaded_files = []
        
        # Verificación opcional (si se usa metadata_map)
        if not isinstance(self.metadata_map, dict):
            raise TypeError("metadata_map debe ser un diccionario.")

def init_session():
    if 'state' not in st.session_state:
        st.session_state.state = SessionState()
        
        # Cargar datos existentes de Supabase a FAISS
        try:
            documentos = st.session_state.state.supabase.client.table('documentos').select("embedding,id").execute().data
            
            for doc in documentos:
                # Convertir cadena a lista de floats
                embedding_list = eval(doc["embedding"])  # ✅ Convertir string a lista
                embedding_array = np.array(embedding_list, dtype=np.float32).reshape(1, -1)
                st.session_state.state.faiss_index.add_with_ids(embedding_array, np.array([doc["id"]]))
                
        except Exception as e:
            st.error(f"Error inicializando FAISS desde Supabase: {str(e)}")


def extract_metadata_from_filename(filename):
    # Ejemplo: "Bravo-Santillana_2021_Tesis.pdf"
    patterns = [
        r"(?P<author>[A-Za-z-]+)_(?P<year>\d{4})_(?P<title>.+?)(\.[^.]+)?$",
        r"(?P<title>.+?)_(?P<author>[A-Za-z-]+)_(?P<year>\d{4})"
    ]

    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return {
                "author": match.group("author").replace("-", " "),
                "year": match.group("year"),
                "title": match.group("title").replace("_", " ")
            }
    
    # Fallback para nombres no estructurados
    return {
        "author": "Desconocido",
        "year": datetime.now().year,
        "title": filename.rsplit('.', 1)[0]
    }

def generate_doc_id(file_name, chunk_index):
    hash_object = hashlib.sha256(f"{file_name}_{chunk_index}".encode())
    return struct.unpack('>q', hash_object.digest()[:8])[0]

def generate_content_hash(texto: str) -> str:
    """Genera un hash único SHA-256 del contenido del texto."""
    texto_limpio = texto.strip().lower().encode('utf-8')
    return hashlib.sha256(texto_limpio).hexdigest()

class PageAwareTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size, chunk_overlap, separators, page_ranges):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        self.page_ranges = page_ranges
        
    def split_text(self, text):
        chunks = super().split_text(text)
        chunk_pages = []
        
        # Mapeo de posiciones a páginas
        page_map = []
        for p_start, p_end, p_num in self.page_ranges:
            page_map.extend([(pos, p_num) for pos in range(p_start, p_end)])
        
        for chunk in chunks:
            start = text.find(chunk)
            end = start + len(chunk)
            pages = set()
            
            # Determinar páginas cubiertas
            for pos in range(start, end):
                if pos < len(page_map):
                    pages.add(page_map[pos][1])
            
            chunk_pages.append({
                "text": chunk,
                "pages": sorted(pages),
                "start_char": start,
                "end_char": end
            })
            
        return chunk_pages
    

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "]
        )

    def extract_text_from_pages(self, file, start_page, end_page):
        """
        Extrae texto de un rango específico de páginas de un archivo PDF.

        Args:
            file: Archivo PDF.
            start_page (int): Número de la primera página (basado en 1).
            end_page (int): Número de la última página (basado en 1).

        Returns:
            str: Texto extraído de las páginas especificadas.
        """
        try:
            file.seek(0)  # Reinicia el puntero del archivo al inicio
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page_num in range(start_page - 1, end_page):  # PyMuPDF usa índice base 0
                page = doc.load_page(page_num)
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error extrayendo texto de páginas {start_page}-{end_page}: {str(e)}")
            return ""

    def process_file(self, file):
        try:
            # Generar hash del contenido completo
            full_content = self._extract_full_content(file)
            doc_hash = generate_content_hash(full_content)
            if st.session_state.state.supabase.is_document_processed(doc_hash):
                st.warning(f"Documento ya procesado: {file.name}")
                return False 

            # Validación inicial del archivo
            if file is None or file.size == 0:
                raise ValueError("Archivo inválido o vacío")

            # Extracción y validación de contenido
            text, page_data = "", []
            
            if file.name.endswith('.pdf'):
                text, page_ranges = self._extract_text_from_pdf(file)
                page_numbers = [p[2] for p in page_ranges]
                
                # Inicializar splitter corregido
                splitter = PageAwareTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", "? ", "! ", " "],
                    page_ranges=page_ranges
                )
                
                chunk_data = splitter.split_text(text)

            elif file.name.endswith('.docx'):
                text = self._extract_text_from_docx(file)
                page_numbers = [1]  # Placeholder para DOCX
                chunk_data = [{"text": chunk, "pages": [1]} for chunk in self.text_splitter.split_text(text)]
            else:
                raise ValueError("Formato de archivo no soportado")

            if not text.strip() or len(text) < 100:
                raise ValueError("Documento sin texto legible o demasiado corto")

            # Insertar en tabla fuentes
            metadata = self._extract_metadata(file)
            fuente_id = st.session_state.state.supabase.insert_fuente(
                metadata={
                    "title": metadata.get("title", file.name),
                    "author": metadata.get("author", "Desconocido"),
                    "paginas_total": len(page_numbers),
                    "anio_publicacion": datetime.now().year,
                    "categoria": metadata.get("categoria", "General")
                },
                content_hash=doc_hash
            )

            # Procesamiento del texto
            if not chunk_data:
                raise ValueError("No se pudieron generar fragmentos válidos")

            embeddings = self._get_embeddings([chunk["text"] for chunk in chunk_data])

            # Procesamiento de cada chunk
            for idx, (chunk_info, embed) in enumerate(zip(chunk_data, embeddings)):
                chunk = chunk_info["text"]
                pages = chunk_info["pages"]
                
                # Generación de metadatos
                doc_id = generate_doc_id(file.name, idx)
                chunk_metadata = {
                    "source": file.name,
                    "pages": f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]),
                    "exact_pages": pages,
                    "start_char": chunk_info.get("start_char", 0),
                    "end_char": chunk_info.get("end_char", 0),
                    "author": metadata.get("author", "Desconocido"),
                    "year": datetime.now().year,
                    "doc_id": doc_id
                }

                # Validación de integridad
                if file.name.endswith('.pdf'):
                    if not self.validate_chunk_integrity(file, chunk, pages):
                        st.warning(f"Chunk {idx} no coincide con las páginas {pages}")

                # Insertar en Supabase
                st.session_state.state.supabase.insert_document(
                    fuente_id=fuente_id,
                    content_hash=generate_content_hash(chunk),
                    contenido=chunk,
                    embedding=embed.tolist(),
                    metadata=chunk_metadata
                )

                # Actualización del almacenamiento
                st.session_state.state.metadata_map[doc_id] = chunk_metadata
                st.session_state.state.document_store[file.name].append(doc_id)

                # Actualización de FAISS
                st.session_state.state.faiss_index.add_with_ids(
                    np.array([embed], dtype=np.float32),
                    np.array([doc_id], dtype=np.int64)
                )

            st.success(f"Documento procesado: {metadata.get('title', file.name)}")
            st.write(f"Chunks generados: {len(chunk_data)} | Páginas procesadas: {len(page_numbers)}")
            return True

        except Exception as e:
            error_details = f"""
            Error procesando {file.name if file else 'archivo'}:
            {str(e)}
            - Tipo archivo: {getattr(file, 'type', 'desconocido')}
            - Tamaño: {getattr(file, 'size', 0)} bytes
            """
            st.error(error_details)
            return False
        
    def _extract_full_content(self, file) -> str:
        """Extrae todo el contenido de un archivo para generar el hash único"""
        try:
            if file.name.endswith('.pdf'):
                text, _ = self._extract_text_from_pdf(file)
                return text
            elif file.name.endswith('.docx'):
                return self._extract_text_from_docx(file)
            else:
                raise ValueError("Formato no soportado")
        except Exception as e:
            st.error(f"Error extrayendo contenido completo: {str(e)}")
            return ""
    def _extract_metadata(self, file):
        if file.name.endswith('.pdf'):
            return self._extract_pdf_metadata(file)
        elif file.name.endswith('.docx'):
            return self._extract_docx_metadata(file)
        else:
            return {
                "categoria": "General",
                "title": file.name,
                "author": "Desconocido",
                "creation_date": "N/A",
                "subject": "",
                "keywords": "",
                "comments": ""
            }
    def _extract_pdf_metadata(self, file):
        """Extrae metadatos de PDFs con soporte para campos vacíos."""
        try:

            file.seek(0)  # ¡Importante! Resetear el puntero del archivo
            doc = fitz.open(stream=file.read(), filetype="pdf")
            metadata = doc.metadata
            
            return {
                "title": metadata.get("title", "Sin título"),
                "author": metadata.get("author", "Desconocido"),
                "creation_date": metadata.get("creationDate", "N/A"),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "comments": metadata.get("comments", "")
            }
        except Exception as e:
            st.error(f"Error extrayendo metadatos PDF: {str(e)}")
            return {
                "categoria": "General",
                "title": "Desconocido",
                "author": "Desconocido",
                "creation_date": "N/A",
                "subject": "",
                "keywords": "",
                "comments": ""
            }

    def _extract_docx_metadata(self, file):
        """Extrae metadatos de archivos DOCX."""
        try:
            doc = Document(file)
            properties = doc.core_properties
            return {
                "title": properties.title or file.name,
                "author": properties.author or "Desconocido",
                "creation_date": str(properties.created) if properties.created else "N/A",
                "subject": properties.subject or "",
                "keywords": properties.keywords or "",
                "comments": properties.comments or ""
            }
        except Exception as e:
            st.error(f"Error extrayendo metadatos DOCX: {str(e)}")
            return {
                "title": file.name,
                "author": "Desconocido",
                "creation_date": "N/A",
                "subject": "",
                "keywords": "",
                "comments": ""
            }

    def _extract_text_from_pdf(self, file):
        try:
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            full_text = []
            page_ranges = []

            current_pos = 0
            for page_num in range(len(doc)):  # Usar índice de página real
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text.append(text)
                end_pos = current_pos + len(text)
                page_ranges.append((
                    current_pos, 
                    end_pos, 
                    page_num + 1  # Páginas base 1 para el usuario
                ))
                current_pos = end_pos + 1  # +1 para separador

            return "\n".join(full_text), page_ranges
        except Exception as e:
            st.error(f"Error extrayendo texto PDF: {str(e)}")
            return "", []
          
    def validate_chunk_integrity(self, file, chunk_text, exact_pages):
        """Verificación precisa usando posiciones caracter"""
        try:
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            full_page_text = ""
            for page_num in exact_pages:
                full_page_text += doc.load_page(page_num - 1).get_text()
                
            return chunk_text in full_page_text
        except Exception as e:
            st.error(f"Error validando chunk: {str(e)}")
            return False
             
    def _extract_text_from_docx(self, file):
        """Extrae texto de DOCX con manejo de formatos complejos."""
        try:
            doc = Document(file)
            full_text = []
            
            # Procesar diferentes elementos del documento
            for element in doc.element.body:
                if isinstance(element, Paragraph):
                    full_text.append(element.text)
                elif isinstance(element, Table):
                    for row in element.rows:
                        for cell in row.cells:
                            full_text.append(cell.text)
            
            return "\n".join(full_text)
        except Exception as e:
            st.error(f"Error leyendo DOCX: {str(e)}")
            return ""
        
    def _get_embeddings(self, chunks):
        """Genera embeddings para una lista de chunks de texto."""
        try:
            # Procesa en lotes para mejorar el rendimiento
            batch_size = 32  # Ajusta según la memoria disponible
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                embeddings.extend(embedder.encode(batch))
            return embeddings
        except Exception as e:
            st.error(f"Error generando embeddings: {str(e)}")
            return []

    def _generate_semantic_tags(self, text):
        """Genera etiquetas semánticas para un texto dado."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            # Lista de stopwords en español
            spanish_stopwords = [
                "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", 
                "con", "no", "una", "su", "al", "es", "lo", "como", "más", "pero", "sus", "le", "ya", 
                "o", "fue", "este", "ha", "sí", "porque", "esta", "son", "entre", "está", "cuando", 
                "muy", "sin", "sobre", "ser", "tiene", "también", "me", "hasta", "hay", "donde", 
                "han", "quien", "están", "estado", "desde", "todo", "nos", "durante", "estados", 
                "todos", "uno", "les", "ni", "contra", "otros", "fueron", "ese", "eso", "había", 
                "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", 
                "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", 
                "cual", "sea", "poco", "ella", "estar", "haber", "estas", "estaba", "estamos", 
                "algunas", "algo", "nosotros"
            ]

            # Configura TfidfVectorizer para español
            vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=10)
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()

            # Obtiene las palabras con mayor puntuación TF-IDF
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(tfidf_scores)[-3:][::-1]  # Top 3 palabras
            tags = [feature_names[i] for i in top_indices]

            return tags
        except Exception as e:
            st.error(f"Error generando etiquetas semánticas: {str(e)}")
            return []
            
# Clase ChatManager
class ChatManager:
    def __init__(self, embedder: SentenceTransformer, supabase_client):
        self.embedder = embedder
        self.supabase = supabase_client  # <-- Recibir y almacenar Supabase
        self.vectorizer = TfidfVectorizer(stop_words='spanish')
        self.similarity_threshold = 0.82
        self.min_chunk_length = 150

# Versión CORREGIDA (app.py)
    def hybrid_search(self, query, top_k=5):
        try:
            # Búsqueda Vectorial
            query_embed = self.embedder.encode([query])[0]
            vector_results = self.supabase.search_documents(query_embed, top_k)
            
            # Búsqueda Léxica - Versión CORREGIDA
            lexical_response = self.supabase.client.rpc(
                'search_lexical',
                {
                    'query_text': query,
                    'top_k': top_k
                }
            ).execute()
            
            # Normalizar resultados léxicos
            lexical_results = []
            for item in lexical_response.data:
                lexical_results.append({
                    "id": item["id"],
                    "contenido": item["contenido"],
                    "metadata": item["metadata"],
                    "similarity": 0.7  # Valor por defecto para compatibilidad
                })
            
            # Combinar y ordenar resultados
            combined_results = vector_results + lexical_results
            return sorted(combined_results, 
                        key=lambda x: x.get('similarity', 0), 
                        reverse=True)[:top_k]
        
        except Exception as e:
            st.error(f"Error en búsqueda híbrida: {str(e)}")
            return []
# Versión CORREGIDA (app.py)
    def generate_response(self, query, results):
        try:
            if not results:
                return "No se encontraron resultados relevantes.", []

            # Paso 1: Procesar metadatos y construir contexto
            source_map = {}  # (autor, título, año) -> {datos}
            context_parts = []
            source_counter = 1  # <-- Nombre de variable corregido
            
            for res in results:  # <-- ¡Atención a la indentación!
                meta = res.get('metadata', {})
                key = (
                    meta.get('author', 'Desconocido'),
                    meta.get('title', 'Sin título'),
                    meta.get('year', 'N/A')
                )
                
                if key not in source_map:
                    # Extraer páginas exactas
                    pages = meta.get('exact_pages', [])
                    if not pages:  # Si no hay páginas, usar las del documento completo
                        pages = list(range(1, meta.get('paginas_total', 1) + 1))
                    
                    source_map[key] = {
                        'number': source_counter,  # <-- Usar source_counter
                        'author': key[0],
                        'title': key[1],
                        'year': key[2],
                        'pages': sorted(set(pages))  # Eliminar duplicados y ordenar
                    }
                    source_counter += 1  # <-- Incrementar contador
                    
                # Construir contexto con cita
                current_source = source_map[key]
                context_parts.append(
                    f"[{current_source['number']}] {current_source['title']} "
                    f"(Páginas {', '.join(map(str, current_source['pages']))})\n"
                    f"{res.get('contenido', '')}"
                )

            # Paso 2: Generar respuesta con el modelo (FUERA del bucle for)
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{
                        "role": "system",
                        "content": "Contexto con citas entre corchetes:\n"  # Parte 1
                                + "\n\n".join(context_parts)  # Parte 2 
                                + "\nResponde usando citas como [n]."  # Parte 3
                    }, {
                        "role": "user",
                        "content": query
                    }]
                }
            )

            # Paso 3: Formatear fuentes finales
            formatted_sources = []
            for key in sorted(source_map.keys(), key=lambda x: source_map[x]['number']):
                data = source_map[key]
                formatted_sources.append({
                    "number": data['number'],
                    "author": key[0],
                    "title": key[1],
                    "year": key[2],
                    "pages": sorted(data['pages'])
                })

            # Procesar respuesta
            if response.status_code == 200:
                response_text = response.json()["choices"][0]["message"]["content"]
                # Convertir [n] a (n)
                response_text = re.sub(r'\[(\d+)\]', r'(\1)', response_text)
                return response_text, formatted_sources
                
            raise Exception(f"API Error: {response.status_code}")
                
        except Exception as e:
            return f"Error: {str(e)}", [] 
        
# Interfaz de usuario mejorada
class DeepSeekUI:
    def __init__(self):
        self.processor = DocumentProcessor()
        # Pasar el cliente Supabase desde el estado de la sesión
        self.chat_manager = ChatManager(
            embedder=embedder,
            supabase_client=st.session_state.state.supabase  # <-- Cliente Supabase
        )
        self.supabase = st.session_state.state.supabase

    def render_sidebar(self):
        with st.sidebar:
            st.header("📥 **Carga de Documentos**")
            
            # 1. Sección principal de carga
            uploaded_files = st.file_uploader(
                "Subir documentos (PDF/DOCX)",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                help="Máximo 200MB por archivo",
                key="main_uploader"
            )
            
            # 2. Botón de procesamiento principal
            if st.button("⚙️ Procesar Documentos", type="primary", key="main_process"):
                if uploaded_files:
                    self._handle_file_processing(uploaded_files)
                else:
                    st.warning("Debes subir al menos un documento")
            
            # 3. Gestión avanzada (SIN EXPANDER ANIDADO)
            with st.expander("🚀 Acciones Avanzadas", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Recarga completa
                    st.subheader("🔄 Recarga Total")
                    st.caption("Elimina todo y carga nuevos documentos")
                    reload_files = st.file_uploader(
                        "Subir para recarga",
                        type=["pdf", "docx"],
                        accept_multiple_files=True,
                        key="reload_uploader"
                    )
                    if st.button("Ejecutar Recarga", type="secondary"):
                        if reload_files:
                            try:
                                st.session_state.state = SessionState()
                                self._handle_file_processing(reload_files)
                                st.success("Recarga completada")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.error("Sube archivos primero")
                
                with col2:
                    # Mantenimiento
                    st.subheader("🧹 Limpieza")
                    st.caption("Acciones de mantenimiento de la base de datos")
                    if st.button("Borrar Todos los Documentos", type="secondary"):
                        try:
                            st.session_state.state.supabase.client.table('documentos').delete().neq('id', 0).execute()
                            st.session_state.state.supabase.client.table('fuentes').delete().neq('id', 0).execute()
                            st.success("Base de datos limpiada")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # 4. Sección de diagnóstico
            with st.expander("📊 Diagnóstico", expanded=False):
                if st.button("Estadísticas de Documentos"):
                    self._show_database_stats()
                
                if st.button("Última Búsqueda"):
                    if hasattr(st.session_state, 'last_search_response'):
                        st.json(st.session_state.last_search_response)
                    else:
                        st.write("Sin registros de búsqueda")
            
            # 5. Fuentes registradas
            with st.expander("📚 Fuentes Actuales", expanded=False):
                fuentes = self.supabase.client.table('fuentes').select("*").order("titulo").execute().data
                for fuente in fuentes:
                    st.write(f"**{fuente['titulo']}**  \n*{fuente['autor']} ({fuente['anio_publicacion']})*")

    def render_chat(self):
        st.title("🧠 Asistente DeepSeek RAG")
        
        # Definir el contenedor PRINCIPAL del chat
        main_chat_container = st.container(height=500)
        
        with main_chat_container:
            # Contenedor interno para historial
            history_container = st.container()
            
            # Contenedor para nuevos mensajes
            message_container = st.container()
            
            # Mostrar historial existente
            with history_container:
                for msg in st.session_state.state.chat_history:
                    self._render_message(msg)
            
            # Procesar nueva consulta
            query = st.chat_input("Escribe tu pregunta...", key="chat_input")
            if query:
                with message_container:
                    self._handle_user_query(query)
                st.rerun()

# Versión CORREGIDA (app.py)
    def _render_message(self, msg):
        with st.container():
            if msg['type'] == 'user':
                st.markdown(f"""
                <div style='
                    margin: 1rem 0; 
                    padding: 1.5rem;
                    background: #e3f2fd;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    position: relative;
                '>
                    <div style='
                        position: absolute;
                        top: -12px;
                        left: 15px;
                        background: #2196F3;
                        color: white;
                        padding: 4px 12px;
                        border-radius: 20px;
                        font-size: 0.85em;
                    '>
                        👤 Tú
                    </div>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
            elif msg['type'] == 'assistant':
                # Mostrar respuesta principal
                st.markdown(msg['content'])
                
                # Mostrar fuentes si existen
                if msg.get('sources'):
                    st.markdown("---")
                    st.markdown("**Fuentes consultadas:**")
                    for source in msg['sources']:
                        st.markdown(f"""
                        <div style='margin: 5px 0; padding: 10px; 
                            background: #f8f9fa; border-radius: 5px;
                            border-left: 3px solid #2c3e50;'>
                            <b>({source['number']})</b> {source['author']} ({source['year']})<br>
                            <i>{source['title']}</i><br>
                            Páginas: {', '.join(map(str, source['pages'])) if source['pages'] else 'N/A'}
                        </div>
                        """, unsafe_allow_html=True)

    def _handle_user_query(self, query):
        try:
            # Obtener resultados de búsqueda
            results = self.chat_manager.hybrid_search(query) or []
            
            # Construir contexto
            context_parts = []
            for res in results:
                source = res['metadata'].get('source', 'Documento desconocido')
                pages = res['metadata'].get('exact_pages', [])
                context_parts.append(
                    f"📚 **{source}** (Páginas {', '.join(map(str, pages))})\n"
                    f"{res['contenido']}\n"
                )
            
            # Generar y mostrar respuesta
            response, sources = self.chat_manager.generate_response(query, results)
            
            # Actualizar historial
            st.session_state.state.chat_history.append({
                'type': 'assistant',
                'content': response,
                'sources': sources
            })
            
        except Exception as e:
            st.error(f"Error procesando consulta: {str(e)}")
        
    def highlight_text(self, text, pages):
        """Resalta contenido con formato académico profesional"""
        pages_str = ", ".join([f"Pág. {p}" for p in sorted(pages)]) if pages else "Pág. N/A"
        
        highlighted = f"""
        <div style='
            border-left: 2px solid #2c3e50;
            margin: 15px 0;
            padding: 12px 20px;
            background: #f9f9f9;
            position: relative;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        '>
            <div style='
                color: #2c3e50;
                font-size: 0.85em;
                font-weight: 600;
                margin-bottom: 8px;
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 5px;
            '>
                <i class="fas fa-book-open" style="margin-right: 7px;"></i>
                Referencia: {pages_str}
            </div>
            <div style='
                color: #34495e;
                line-height: 1.6;
                font-size: 0.95em;
                text-align: justify;
            '>
                {text}
            </div>
        </div>
        """
        return highlighted    

    def _validate_response(self, response, context):
        validation_prompt = f"""
        Utilice solo los siguientes fragmentos de contexto para responder a la pregunta al final. Si la respuesta, no esta ahi, simplemente diga que no la sabe, no intente inventar una respuesta.
        
        **Respuesta:**
        {response}
        
        **Contexto:**
        {context}
        
        Proporciona una validación en formato JSON con:
        - score (1-5)
        - valid (true/false)
        - reasons (lista de razones)
        """
        
        validation = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user",
                    "content": validation_prompt
                }],
                "response_format": {"type": "json_object"}
            }
        )
        if validation.status_code == 200:
            return json.loads(validation.json()["choices"][0]["message"]["content"])
        else:
            return {"score": 0, "valid": False, "reasons": ["Error en la validación"]}

    def _show_database_stats(self):
        """Muestra estadísticas clave de la base de datos"""
        try:
            # Acceder a través de self.supabase
            docs_count = self.supabase.client.table('documentos').select("count", count='exact').execute().count
            fuentes_count = self.supabase.client.table('fuentes').select("count", count='exact').execute().count
            
            st.subheader("📊 Estadísticas del Sistema")
            col1, col2 = st.columns(2)
            col1.metric("Documentos (Chunks)", docs_count)
            col2.metric("Fuentes Únicas", fuentes_count)
            
        except Exception as e:
            st.error(f"Error obteniendo estadísticas: {str(e)}")

    def _handle_file_processing(self, files):
        """Maneja el procesamiento de archivos en lote con seguimiento visual"""
        # Inicializar lista de archivos subidos
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        # Configurar elementos de UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, file in enumerate(files):
                # Actualizar UI
                status_text.markdown(f"""
                    **Procesando:** `{file.name}`  
                    📏 Tamaño: {file.size/1e6:.1f} MB  
                    🔖 Tipo: {file.type.split('/')[-1].upper()}
                """)
                
                # Procesar archivo
                success = self.processor.process_file(file)
                
                # Manejar resultados
                if success:
                    st.session_state.uploaded_files.append(file.name)
                    st.success(f"✅ {file.name} procesado correctamente")
                else:
                    st.warning(f"⚠️ {file.name} tuvo problemas en el procesamiento")
                
                # Actualizar barra de progreso
                progress_bar.progress((i + 1) / len(files))
                
        except Exception as e:
            st.error(f"""
                **Error crítico:**  
                ```python
                {str(e)}
                ```
                🛠️ **Solución:**  
                - Verifique el formato del archivo  
                - Asegúrese de no subir archivos protegidos con contraseña  
                - Revise los logs técnicos
            """)
            st.exception(e)  # Mostrar traceback completo para depuración
            
        finally:
            # Limpiar elementos de UI
            progress_bar.empty()
            status_text.empty()

# Inicialización y ejecución
if __name__ == "__main__":
    init_session()
    ui = DeepSeekUI()
    ui.render_sidebar()
    ui.render_chat()
