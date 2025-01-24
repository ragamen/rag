from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Tried to instantiate class '__path__._path'")
import os
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
from io import BytesIO
from functools import lru_cache
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import os
from dotenv import load_dotenv
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# st.error(f"Clave Api key {DEEPSEEK_API_KEY}")

# Configuración inicial
embedder = SentenceTransformer('all-MiniLM-L6-v2')


class SessionState:
    def __init__(self):
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        self.faiss_index = faiss.IndexIDMap(index)
        self.metadata_map = {}
        self.document_store = defaultdict(list)
        self.chat_history = []
        self.uploaded_files = []
        self.current_page = None  # Nueva variable para seguimiento de página
        # Verificación inicial
        if not isinstance(self.metadata_map, dict):
            raise TypeError("metadata_map debe ser un diccionario.")

def init_session():
    if 'state' not in st.session_state:
        st.session_state.state = SessionState()



def extract_metadata_from_filename(filename):
    # Ejemplo: "Bravo-Santillana_2021_Tesis.pdf"
    pattern = r"^(?P<author>[^_]+)_(?P<year>\d{4})_(?P<title>.+)\.pdf$"
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    return {"author": "Desconocido", "year": "N/A", "title": filename}

def generate_doc_id(file_name, chunk_index):
    hash_object = hashlib.sha256(f"{file_name}_{chunk_index}".encode())
    return struct.unpack('>q', hash_object.digest()[:8])[0]

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
            if file is None or file.size == 0:
                st.error(f"Archivo inválido: {file.name}")
                return False

            metadata = self._extract_metadata(file)
            text, page_numbers = self._extract_text_from_pdf(file)  # Extrae texto y números de página
            chunks = self.text_splitter.split_text(text)
            
            embeddings = self._get_embeddings(chunks)

            for idx, (chunk, embed) in enumerate(zip(chunks, embeddings)):
                # Determina las páginas correspondientes al chunk
                start_page = page_numbers[0] if page_numbers else None
                end_page = page_numbers[-1] if page_numbers else None

                # Verificar que el chunk esté contenido en las páginas
                if start_page and end_page:
                    page_text = self.extract_text_from_pages(file, start_page, end_page)
                    if chunk not in page_text:
                        st.warning(f"El chunk {idx} no está completamente contenido en las páginas {start_page}-{end_page}.")

                doc_id = generate_doc_id(file.name, idx)
                
                chunk_metadata = {
                    "doc_id": doc_id,
                    "content": chunk,  # Texto sin números de página
                    "embedding": embed,
                    "source": file.name,
                    "timestamp": datetime.now().isoformat(),
                    "semantic_tags": self._generate_semantic_tags(chunk),
                    "metadata": {
                        **metadata,
                        "pages": f"{start_page}-{end_page}" if start_page != end_page else str(start_page)
                    }
                }
                
                # Verificación de la estructura de chunk_metadata
                required_keys = ["doc_id", "content", "embedding", "source", "timestamp", "semantic_tags", "metadata"]
                if not all(key in chunk_metadata for key in required_keys):
                    raise ValueError(f"Falta una clave requerida en chunk_metadata: {chunk_metadata}")
                
                st.session_state.state.metadata_map[doc_id] = chunk_metadata
                st.session_state.state.document_store[file.name].append(doc_id)

                # Actualizar FAISS
                embeddings_array = np.array([embed], dtype=np.float32)
                ids_array = np.array([doc_id], dtype=np.int64)
                st.session_state.state.faiss_index.add_with_ids(embeddings_array, ids_array)

            st.success(f"Documento procesado: {metadata['title']}")
            st.write(f"Páginas procesadas: {len(page_numbers)}")
            return True
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
            return False

    def _extract_metadata(self, file):
        if file.name.endswith('.pdf'):
            return self._extract_pdf_metadata(file)
        elif file.name.endswith('.docx'):
            return self._extract_docx_metadata(file)
        else:
            return {
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
        """Extrae texto de un archivo PDF y registra los números de página en los metadatos."""
        try:
            file.seek(0)  # Reinicia el puntero del archivo al inicio
            doc = fitz.open(stream=file.read(), filetype="pdf")
            full_text = []
            page_numbers = []  # Almacena los números de página

            for page_num, page in enumerate(doc):
                text = page.get_text()
                full_text.append(text)
                page_numbers.append(page_num + 1)  # Los números de página comienzan en 1

            return "\n".join(full_text), page_numbers
        except Exception as e:
            st.error(f"Error extrayendo texto PDF: {str(e)}")
            return "", []

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
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_threshold = 0.82  # Más estricto
        self.min_chunk_length = 150  # Ignorar chunks cortos    

    def hybrid_search(self, query, top_k=5):
        try:
            # Búsqueda vectorial con FAISS
            query_embed = embedder.encode([query])[0]
            D, I = st.session_state.state.faiss_index.search(
                np.array([query_embed], dtype=np.float32), 
                top_k * 2
            )
            faiss_ids = I[0].tolist()

            # Búsqueda léxica con TF-IDF
            corpus = []
            id_map = []  # Mapeo de índices a IDs reales
            for doc_id, metadata in st.session_state.state.metadata_map.items():
                # Verificación de la estructura de metadata
                required_keys = ["content", "source", "metadata", "semantic_tags"]
                if not all(key in metadata for key in required_keys):
                    raise ValueError(f"Falta una clave requerida en metadata: {metadata}")
                
                corpus.append(metadata['content'])
                id_map.append(doc_id)
            
            if corpus:
                tfidf_matrix = self.vectorizer.fit_transform(corpus)
                query_vec = self.vectorizer.transform([query])
                doc_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
                lexical_ids = [id_map[i] for i in np.argsort(doc_scores)[-top_k * 2:][::-1]]
            else:
                lexical_ids = []

            # Combinar y priorizar resultados
            combined_ids = list(set(faiss_ids + lexical_ids))
            
            # Recuperar metadatos y ordenar por relevancia
            results = []
            for doc_id in combined_ids:
                if doc_id in st.session_state.state.metadata_map:
                    metadata = st.session_state.state.metadata_map[doc_id]
                    results.append(metadata)
            
            # Ordenar por puntaje combinado (simulación)
            results = sorted(results, 
                key=lambda x: len(x['semantic_tags']), 
                reverse=True
            )[:top_k]

            return results
        except Exception as e:
            st.error(f"Error en búsqueda: {str(e)}")
            return []  
        
    def generate_response(self, query, context, sources):
        try:
            # Depuración: Verifica la estructura de `sources`
            # st.write("Estructura de sources:", sources)  # Muestra la estructura de sources
            
            # Asegúrate de que `sources` sea una lista de diccionarios
            if not isinstance(sources, list) or not all(isinstance(src, dict) for src in sources):
                raise ValueError("La estructura de 'sources' es inválida. Debe ser una lista de diccionarios.")
            
            # Agrupar fuentes por autor, título y año
            grouped_sources = defaultdict(list)
            for src in sources:
                metadata = extract_metadata_from_filename(src.get("source", "Desconocido"))
                key = (metadata["author"], metadata["title"], metadata["year"])
                grouped_sources[key].append(src)
            
            # Formatear las fuentes agrupadas
            formatted_sources = []
            for (author, title, year), src_list in grouped_sources.items():
                pages = set()
                tags = set()
                for src in src_list:
                    pages.add(src.get("metadata", {}).get("pages", "N/A"))
                    tags.update(src.get("semantic_tags", []))
                
                formatted_sources.append(
                    f"{author} ({year}). {title}. "
                    f"Páginas: {', '.join(sorted(pages))}. "
                    f"Etiquetas: {', '.join(sorted(tags)[:3])}"
                )
            
            # Generar la respuesta usando la API de DeepSeek
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
                        "content": f"Contexto con páginas:\n{context}\nResponde profesionalmente citando páginas."
                    }, {
                        "role": "user",
                        "content": query
                    }]
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"], formatted_sources
            else:
                raise Exception(f"API Error: {response.status_code}")
                
        except Exception as e:
            return f"Error: {str(e)}", []        
        
        
# Interfaz de usuario mejorada
class DeepSeekUI:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.chat_manager = ChatManager()
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("⚙️ Configuración")
            
            # Botón para cambiar entre modo claro y oscuro
            dark_mode = st.checkbox("Modo Oscuro")
            
            if dark_mode:
                st.markdown("""
                <style>
                    .main { background-color: #1e1e1e; color: white; }
                    .stButton>button { background-color: #4CAF50; color: white; }
                    .response-box { 
                        border: 2px solid #4CAF50;
                        border-radius: 5px;
                        padding: 20px;
                        margin: 10px 0;
                        background-color: #2e2e2e;
                    }
                    .reference-item { 
                        margin: 5px 0;
                        padding: 10px;
                        background-color: #3e3e3e;
                        border-radius: 3px;
                    }
                </style>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <style>
                    .main { background-color: #f0f2f6; color: black; }
                    .stButton>button { background-color: #4CAF50; color: white; }
                    .response-box { 
                        border: 2px solid #4CAF50;
                        border-radius: 5px;
                        padding: 20px;
                        margin: 10px 0;
                        background-color: #ffffff;
                    }
                    .reference-item { 
                        margin: 5px 0;
                        padding: 10px;
                        background-color: #e8f5e9;
                        border-radius: 3px;
                    }
                </style>
                """, unsafe_allow_html=True)
            
            with st.expander("📤 Gestión de Documentos"):
                uploaded_files = st.file_uploader(
                    "Subir documentos (PDF/DOCX)",
                    type=["pdf", "docx"],
                    accept_multiple_files=True
                )
                if st.button("Procesar Documentos", type="primary"):
                    if uploaded_files:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file in enumerate(uploaded_files):
                            status_text.text(f"Procesando {i + 1}/{len(uploaded_files)}: {file.name}")
                            if file.name not in st.session_state.state.uploaded_files:
                                try:
                                    success = self.processor.process_file(file)
                                    if success:
                                        st.session_state.state.uploaded_files.append(file.name)
                                    else:
                                        st.warning(f"No se pudo procesar el archivo: {file.name}")
                                except Exception as e:
                                    st.error(f"Error procesando archivo '{file.name}': {str(e)}")
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        status_text.text("Procesamiento completado.")
                        st.success("Todos los documentos han sido procesados.")
                    else:
                        st.warning("No se han subido archivos.")
            
            # Verificación de metadata_map
            #"""
            #with st.expander("🔍 Verificar metadata_map"):
            #    if st.button("Verificar Estructura de metadata_map"):
            #        if not isinstance(st.session_state.state.metadata_map, dict):
            #            st.error("metadata_map no es un diccionario.")
            #        else:
            #            st.write("Estructura de metadata_map:", st.session_state.state.metadata_map)
            #"""

            with st.expander("🔍 Opciones de Búsqueda"):
                self.search_type = st.radio(
                    "Modo de búsqueda:",
                    ["Híbrida", "Vectorial", "Semántica"],
                    index=0
                )
                
                self.creativity = st.slider(
                    "Nivel de creatividad:",
                    min_value=0.0, max_value=1.0, value=0.5,
                    help="Controla el balance entre precisión y originalidad"
                )
        
    def render_chat(self):
        st.title("🧠 Asistente DeepSeek RAG")
        
        # Historial de chat
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.state.chat_history:
                self._render_message(msg)
          
            # Entrada de usuario
        query = st.chat_input("Escribe tu pregunta...")
        if query:
            self._handle_user_query(query)
        
    def _render_message(self, msg):
        if msg['type'] == 'user':
            st.markdown(f"""
            <div style="margin: 1rem 0; padding: 1rem; 
                        background: #e3f2fd; border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <strong>👤 Tú:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
            
        elif msg['type'] == 'assistant':
            with st.expander("💡 Respuesta Completa", expanded=True):
                st.markdown(f"""
                <div style="margin: 0.5rem 0; padding: 1rem;
                            background: #f5f5f5; border-radius: 10px;">
                    <strong>🤖 Asistente:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                    
                if 'references' in msg:
                    st.markdown("**🔍 Fuentes Relacionadas:**")
                    for ref in msg['references']:
                        st.markdown(f"""
                        <div class="reference-item">
                            📌 {ref}
                        </div>
                        """, unsafe_allow_html=True)
    
    def _handle_user_query(self, query):
        st.session_state.state.chat_history.append({
            'type': 'user',
            'content': query
        })
        
        with st.spinner("🔍 Buscando información relevante..."):
            results = self.chat_manager.hybrid_search(query)
            
            # Construir el contexto y las fuentes
            context = "\n\n".join(
                f"[Fuente: {res['source']}]\n{res['content']}" 
                for res in results
            )
            
            # Asegurarse de que `sources` sea una lista de diccionarios
            sources = []
            for res in results:
                if isinstance(res, dict):  # Verificar que cada resultado sea un diccionario
                    sources.append({
                        "source": res.get("source", "Desconocido"),
                        "metadata": res.get("metadata", {}),
                        "semantic_tags": res.get("semantic_tags", [])
                    })
            
        with st.spinner("💡 Generando respuesta..."):
            response, response_sources = self.chat_manager.generate_response(query, context, sources)
            
            st.session_state.state.chat_history.append({
                'type': 'assistant',
                'content': response,
                'references': response_sources,  # Incluye las fuentes en la respuesta
                'validation': self._validate_response(response, context)
            })
        
        st.rerun()  
        
         
    def _validate_response(self, response, context):
        validation_prompt = f"""
        Utilice los siguientes fragmentos de contexto para responder a la pregunta al final. Si no sabe la respuesta, simplemente diga que no la sabe, no intente inventar una respuesta.
        
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

# Inicialización y ejecución
if __name__ == "__main__":
    init_session()
    ui = DeepSeekUI()
    ui.render_sidebar()
    ui.render_chat()
