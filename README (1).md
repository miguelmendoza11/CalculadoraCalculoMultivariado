# Calculadora de Cálculo Multivariable

Aplicación web desarrollada en **Streamlit** para visualizar, calcular e interpretar conceptos de derivadas parciales e integrales múltiples.

## 🚀 Características

- Visualización 3D interactiva de funciones de dos variables
- Cálculo de dominio y rango
- Derivadas parciales y gradientes
- Optimización con y sin restricciones (Multiplicadores de Lagrange)
- Integración doble para cálculo de volúmenes
- **Aplicaciones prácticas generadas por IA (Google Gemini)**
- Interfaz moderna y responsive

## 📋 Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🔧 Instalación

1. Instalar las dependencias:
```bash
pip install streamlit numpy sympy plotly scipy requests
```

O con el archivo requirements:
```bash
pip install -r requirements.txt
```

2. Configurar la API de Google Gemini (opcional):
   
   Para obtener aplicaciones prácticas de las funciones mediante IA:
   
   a) Obtener una API key gratuita en: https://aistudio.google.com/app/apikey
   
   b) Configurar la variable de entorno:
   
   **Windows:**
   ```bash
   set GEMINI_API_KEY=tu_api_key_aqui
   ```
   
   **Linux/Mac:**
   ```bash
   export GEMINI_API_KEY=tu_api_key_aqui
   ```
   
   **O editar directamente en app_streamlit.py línea 13:**
   ```python
   GEMINI_API_KEY = 'tu_api_key_aqui'
   ```

## ▶️ Ejecución

**Versión Streamlit (RECOMENDADA):**
```bash
streamlit run app_streamlit.py
```

**Versión Flask (alternativa):**
```bash
python app.py
```

Abre el navegador en la URL que se muestra en la terminal (generalmente `http://localhost:8501` para Streamlit)

## 📖 Uso

### Ingresar una función
- Escribir la función usando sintaxis de Python
- Ejemplos: `x**2 + y**2`, `sin(x)*cos(y)`, `exp(-(x**2 + y**2))`

### Visualización 3D
- Ajustar los rangos de x e y en el sidebar
- Las gráficas son interactivas (rotar, zoom, pan)
- Estilo GeoGebra con colores suaves y curvas de nivel

### Derivadas Parciales
- Se calculan automáticamente ∂f/∂x y ∂f/∂y
- Se muestra el gradiente en el punto especificado
- Formato LaTeX para mejor visualización

### Optimización
- Encuentra puntos críticos automáticamente
- Ingresar una restricción opcional para usar Lagrange
- Formato: `x**2 + y**2 - 4`

### Integración
- Especificar los límites de integración
- Calcula el volumen bajo la superficie
- Integración simbólica y numérica automática

## 🎓 Funciones de Ejemplo

- **Paraboloide**: `x**2 + y**2`
- **Silla de montar**: `x**2 - y**2`
- **Ondas**: `sin(x)*cos(y)`
- **Gaussiana**: `exp(-(x**2 + y**2))`
- **Cono**: `sqrt(x**2 + y**2)`

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework web interactivo
- **SymPy**: Cálculo simbólico
- **NumPy**: Cálculo numérico
- **Plotly**: Visualización 3D interactiva
- **SciPy**: Optimización e integración numérica
- **Google Gemini API**: Generación de aplicaciones prácticas con IA

## 👥 Proyecto Final de Cálculo Multivariable

Desarrollado como proyecto final del curso de Cálculo Multivariable.
