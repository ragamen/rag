class UNESCOMetadataValidator:
    # Taxonomía oficial UNESCO (versión simplificada)
    UNESCO_CATEGORIES = {
        "11": "Lógica",
        "12": "Matemáticas",
        "21": "Astronomía y Astrofísica",
        "22": "Física",
        "31": "Ciencias Agrícolas",
        "32": "Ciencias Médicas",
        "53": "Ciencias Económicas",
        "54": "Sociología",
        "55": "Derecho",
        "56": "Ciencias Políticas",
        "58": "Pedagogía",
        "63": "Agronomía",
        "72": "Ética"
    }

    @classmethod
    def validate_metadata(cls, metadata: dict) -> dict:
        """Valida y normaliza los metadatos según estándares UNESCO"""
        errors = []
        
        # Validación de campos obligatorios
        required_fields = ['title', 'author', 'year']
        for field in required_fields:
            if not metadata.get(field):
                errors.append(f"Campo requerido faltante: {field}")
        
        # Normalización de categoría UNESCO
        category = metadata.get('categoria', '')
        if category not in cls.UNESCO_CATEGORIES.values():
            suggestions = cls.suggest_category(category)
            errors.append(f"Categoría inválida. Sugerencias: {', '.join(suggestions)}")
        
        return {
            "is_valid": len(errors) == 0,
            "normalized_metadata": {
                "title": metadata['title'].strip().title(),
                "author": metadata['author'].strip().title(),
                "year": int(metadata['year']),
                "unesco_category": cls.match_unesco_category(metadata['categoria'])
            },
            "errors": errors
        }

    @classmethod
    def match_unesco_category(cls, user_input: str) -> str:
        """Busca coincidencia más cercana en la taxonomía UNESCO"""
        user_input = user_input.lower()
        for code, category in cls.UNESCO_CATEGORIES.items():
            if user_input in category.lower():
                return f"{code} - {category}"
        return "99 - Otras categorías"
