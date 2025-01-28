import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
import tempfile
import os
import spacy
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Cargar el modelo de lenguaje de spaCy
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    st.error("El modelo de lenguaje en español no está instalado. Ejecuta 'python -m spacy download es_core_news_sm' para instalarlo.")
    st.stop()

def extract_keywords(text, num_keywords=10):
    """
    Extrae palabras clave de un texto utilizando spaCy.
    """
    # Procesar el texto con spaCy
    doc = nlp(text)

    # Filtrar palabras clave (sustantivos, adjetivos, verbos)
    keywords = [
        token.text.lower() for token in doc
        if token.pos_ in ["NOUN", "ADJ", "VERB"]  # Filtrar por tipo de palabra
        and token.text.lower() not in nlp.Defaults.stop_words  # Excluir palabras vacías
        and token.text not in string.punctuation  # Excluir puntuación
    ]

    # Contar la frecuencia de las palabras clave
    keyword_freq = Counter(keywords)

    # Obtener las palabras clave más comunes
    most_common_keywords = keyword_freq.most_common(num_keywords)

    # Devolver solo las palabras clave (sin la frecuencia)
    return [keyword for keyword, freq in most_common_keywords]

def generate_summary(text, num_sentences=3):
    """
    Genera una descripción breve del texto extrayendo las oraciones más relevantes.
    """
    # Dividir el texto en oraciones
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Calcular la importancia de las oraciones usando TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Seleccionar las oraciones más importantes
    top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    summary = " ".join([sentences[i] for i in top_sentence_indices])

    return summary

def update_pdf_metadata(pdf_path, metadata):
    """
    Actualiza los metadatos de un PDF.
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Copiar todas las páginas al nuevo PDF
    for page in reader.pages:
        writer.add_page(page)

    # Actualizar metadatos
    writer.add_metadata(metadata)

    # Guardar el PDF con los nuevos metadatos en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        output_path = tmp_file.name
        writer.write(tmp_file)

    return output_path

def main():
    st.title("Edición de Metadatos de PDF")

    # Dropdown de categorías según normas internacionales (UNESCO)
    categorias = [
        "Generalidades",
        "Filosofía",
        "Psicología",
        "Religión, teología",
        "Sociología",
        "Estadística",
        "Ciencias políticas",
        "Economía política",
        "Derecho",
        "Administración pública",
        "Previsión, asistencia social, seguros",
        "Arte y ciencia militar",
        "Enseñanza, educación",
        "Comercio, comunicaciones, transportes",
        "Etnografía, usos y costumbres, folklore",
        "Lingüística, filología",
        "Matemáticas",
        "Ciencias naturales",
        "Ciencias médicas, higiene pública",
        "Ingeniería, tecnología, industrias, artes y oficios",
        "Agricultura, silvicultura, ganadería, caza, pesca",
        "Economía doméstica",
        "Organización, administración y técnica del comercio",
        "Comunicaciones, transportes",
        "Urbanismo, arquitectura",
        "Artes plásticas, oficios artísticos",
        "Fotografía, film, cinematografía",
        "Música, cinematografía, teatro, radio televisión",
        "Recreos, pasatiempos, juegos, deportes",
        "Literatura: Historia y crítica literarias",
        "Literatura: Textos Literarios",
        "Geografía, viajes",
        "Historia, biografía",
        "Libros de texto",
        "Libros para niños"
    ]

    # Cargar PDF
    uploaded_file = st.file_uploader("Cargar PDF", type="pdf")
    if uploaded_file is not None:
        # Guardar el archivo cargado en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Leer metadatos actuales
        reader = PdfReader(pdf_path)
        metadata = reader.metadata

        # Mostrar metadatos actuales
        st.subheader("Metadatos Actuales")
        st.write(f"**Título:** {metadata.get('/Title', 'Sin título')}")
        st.write(f"**Autor:** {metadata.get('/Author', 'Desconocido')}")
        st.write(f"**Categoría:** {metadata.get('/Category', 'Sin categoría')}")
        st.write(f"**Descripción:** {metadata.get('/Subject', 'Sin descripción')}")
        st.write(f"**Palabras clave:** {metadata.get('/Keywords', 'N/A')}")
        st.write(f"**Fecha de creación:** {metadata.get('/CreationDate', 'N/A')}")

        # Extraer texto del PDF
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Extraer palabras clave automáticamente
        st.subheader("Palabras Clave Extraídas")
        keywords = extract_keywords(text, num_keywords=10)
        st.write(", ".join(keywords))

        # Generar descripción breve
        st.subheader("Descripción Breve Generada")
        summary = generate_summary(text)
        st.write(summary)

        # Campos para editar metadatos
        st.subheader("Editar Metadatos")
        new_title = st.text_input("Título del PDF:", value=metadata.get('/Title', ''))
        new_author = st.text_input("Autor del PDF:", value=metadata.get('/Author', ''))
        new_category = st.selectbox("Seleccione una categoría:", categorias, index=categorias.index(metadata.get('/Category', 'Generalidades')))
        new_keywords = st.text_input("Palabras clave (separadas por comas):", value=", ".join(keywords))
        new_description = st.text_area("Descripción del PDF:", value=summary)
        new_creation_date = st.date_input("Fecha de creación:", value=datetime.now())

        # Botón para guardar cambios
        if st.button("Guardar Metadatos"):
            # Crear diccionario con los nuevos metadatos
            new_metadata = {
                '/Title': new_title,
                '/Author': new_author,
                '/Category': new_category,
                '/Keywords': new_keywords,
                '/Subject': new_description,
                '/CreationDate': new_creation_date.strftime("D:%Y%m%d%H%M%S")
            }

            # Actualizar metadatos del PDF
            updated_pdf_path = update_pdf_metadata(pdf_path, new_metadata)

            # Mostrar mensaje de éxito
            st.success("Metadatos guardados correctamente!")

            # Opción para descargar el PDF actualizado
            with open(updated_pdf_path, "rb") as file:
                st.download_button(
                    label="Descargar PDF actualizado",
                    data=file,
                    file_name=f"updated_{uploaded_file.name}",
                    mime="application/pdf"
                )

            # Eliminar archivos temporales
            os.unlink(pdf_path)
            os.unlink(updated_pdf_path)

if __name__ == "__main__":
    main()