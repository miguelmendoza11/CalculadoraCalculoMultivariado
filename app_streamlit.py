import streamlit as st
import numpy as np
import sympy as sp
from sympy import symbols, lambdify, sympify, diff, integrate, solve
import plotly.graph_objects as go
from scipy.optimize import minimize
import requests
import os

# Configuración de página
st.set_page_config(
    page_title="Calculadora de Cálculo Multivariable",
    page_icon="📊",
    layout="wide"
)

# Configuración de Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDjcmpdzo4HiOwi7Ct-NF5m2PUqe0uWTFc')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #cbd5e0 0%, #a0aec0 100%);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
    }
    h1 {
        color: #4a5568 !important;
        text-align: center;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 25px;
        background-color: white;
        border-radius: 8px;
        color: #333;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #718096 !important;
        color: white !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #cbd5e0 0%, #a0aec0 100%);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #2d3748 !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        font-weight: 700 !important;
        border: 2px solid #718096 !important;
        text-align: center !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #718096 !important;
        color: white !important;
        border: 2px solid #718096 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border: 2px solid rgba(45, 55, 72, 0.2);
    }
    [data-testid="stSidebar"] input::placeholder {
        color: #999 !important;
    }
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #333 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #666 !important;
        font-weight: 600 !important;
    }
    /* Texto general */
    p, span, div {
        color: #333 !important;
    }
    /* Títulos y subtítulos */
    h2, h3, h4 {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("<h1 style='color: #4a5568; text-align: center; font-size: 2.5rem;'>📊 Calculadora de Cálculo Multivariable</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.1rem; font-weight: 500;'>Visualización, Cálculo e Interpretación de Funciones Multivariables</p>", unsafe_allow_html=True)
st.markdown("---")

# Función para obtener aplicaciones prácticas
def get_function_application(function_str):
    # Diccionario de aplicaciones predefinidas para funciones comunes
    aplicaciones_comunes = {
        "x**2 + y**2": "Esta función representa un paraboloide de revolución. Se utiliza en física para modelar antenas parabólicas y telescopios reflectores, en ingeniería civil para diseñar cúpulas y estructuras, y en economía para funciones de costo cuadrático.",
        "x**2 - y**2": "Esta función representa una silla de montar (punto silla). Se usa en teoría de juegos para modelar situaciones de equilibrio, en física para describir campos electromagnéticos, y en arquitectura para diseñar techos hiperbólicos como el Palacio de los Deportes en México.",
        "sin(x)*cos(y)": "Esta función periódica se aplica en el análisis de ondas bidimensionales, diseño de patrones de interferencia en óptica, procesamiento de señales digitales, y en el modelado de vibraciones en membranas y placas.",
        "exp(-(x**2 + y**2))": "Esta función gaussiana bidimensional se utiliza en procesamiento de imágenes para suavizado (filtro gaussiano), en estadística para distribuciones normales bivariadas, en física cuántica para funciones de onda, y en aprendizaje automático para funciones de activación."
    }

    # Buscar si hay una aplicación predefinida
    if function_str in aplicaciones_comunes:
        return aplicaciones_comunes[function_str]

    # Si no está en el diccionario, intentar con la API
    try:
        headers = {'Content-Type': 'application/json'}
        prompt = f"""Describe brevemente (máximo 2-3 oraciones) una aplicación práctica real de la función matemática f(x,y) = {function_str} en ingeniería, física, economía o ciencias.
        Sé específico y conciso. No uses formato markdown ni asteriscos."""

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0]:
                    if 'parts' in result['candidates'][0]['content']:
                        text = result['candidates'][0]['content']['parts'][0]['text']
                        return text.strip()
    except:
        pass

    # Fallback genérico sin mostrar errores
    return "Esta función tiene múltiples aplicaciones en modelado matemático, análisis científico y problemas de optimización en ingeniería y ciencias aplicadas."

# Sidebar - Entrada de datos
with st.sidebar:
    st.markdown("<h2 style='color: #2d3748; font-size: 1.5rem;'>🔢 Función</h2>", unsafe_allow_html=True)
    
    func_str = st.text_input(
        "Ingrese la función f(x,y):",
        value="x**2 + y**2",
        help="Ejemplo: x**2 + y**2, sin(x)*cos(y), exp(-(x**2 + y**2))",
        label_visibility="collapsed"
    )
    
    st.markdown("<p style='color: rgba(45, 55, 72, 0.8); font-size: 0.85rem; margin-top: -10px;'>Ejemplo: x**2 + y**2, sin(x)*cos(y)</p>", unsafe_allow_html=True)

    st.markdown("<p style='color: #2d3748; font-weight: 600; font-size: 1.1rem; margin-top: 20px; margin-bottom: 10px;'>Funciones de ejemplo:</p>", unsafe_allow_html=True)
    
    button_style = """
        <style>
        /* Estilos específicos para botones de funciones de ejemplo */
        [data-testid="stSidebar"] div.stButton > button,
        div.stButton > button {
            background-color: #f8f9fa !important;
            color: #212529 !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            border: 2px solid #718096 !important;
            padding: 0.7rem 1rem !important;
            width: 100% !important;
            margin-bottom: 0.5rem !important;
            border-radius: 8px !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            transition: all 0.3s ease !important;
            text-align: center !important;
            line-height: 1.4 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        [data-testid="stSidebar"] div.stButton > button:hover,
        div.stButton > button:hover {
            background-color: #718096 !important;
            color: white !important;
            border: 2px solid #718096 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(113, 128, 150, 0.3) !important;
        }
        [data-testid="stSidebar"] div.stButton > button:focus,
        div.stButton > button:focus {
            color: #212529 !important;
            background-color: #f8f9fa !important;
            border: 2px solid #718096 !important;
            outline: none !important;
        }
        [data-testid="stSidebar"] div.stButton > button:active,
        div.stButton > button:active {
            color: white !important;
            background-color: #4a5568 !important;
            border: 2px solid #4a5568 !important;
        }
        /* Forzar el color del texto en todos los elementos internos */
        [data-testid="stSidebar"] div.stButton > button *,
        div.stButton > button * {
            color: inherit !important;
            font-weight: inherit !important;
        }
        /* Selector más específico para el texto del botón */
        [data-testid="stSidebar"] div.stButton > button p,
        [data-testid="stSidebar"] div.stButton > button span,
        [data-testid="stSidebar"] div.stButton > button div,
        div.stButton > button p,
        div.stButton > button span,
        div.stButton > button div {
            color: #212529 !important;
            font-weight: 700 !important;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Paraboloide"):
            func_str = "x**2 + y**2"
            st.rerun()
        if st.button("Silla de montar"):
            func_str = "x**2 - y**2"
            st.rerun()
    with col2:
        if st.button("Seno-Coseno"):
            func_str = "sin(x)*cos(y)"
            st.rerun()
        if st.button("Gaussiana"):
            func_str = "exp(-(x**2 + y**2))"
            st.rerun()
    
    st.markdown("<h2 style='color: #2d3748; font-size: 1.5rem; margin-top: 30px;'>📐 Rango de visualización</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("x min:", value=-5.0, step=0.5, label_visibility="visible")
        y_min = st.number_input("y min:", value=-5.0, step=0.5, label_visibility="visible")
    with col2:
        x_max = st.number_input("x max:", value=5.0, step=0.5, label_visibility="visible")
        y_max = st.number_input("y max:", value=5.0, step=0.5, label_visibility="visible")
    
    st.markdown("<h2 style='color: #2d3748; font-size: 1.5rem; margin-top: 30px;'>🎯 Punto de evaluación</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        point_x = st.number_input("x₀:", value=1.0, step=0.1, label_visibility="visible")
    with col2:
        point_y = st.number_input("y₀:", value=1.0, step=0.1, label_visibility="visible")
    
    st.markdown("<br>", unsafe_allow_html=True)
    calculate_btn = st.button("🚀 Calcular y Visualizar", type="primary", width="stretch")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualización 3D", "∂ Derivadas", "🎯 Optimización", "∫ Integración"])

# Variables simbólicas
x, y = symbols('x y')

# Inicializar session_state si es necesario
if 'calculated_data' not in st.session_state:
    st.session_state.calculated_data = None

# Calcular cuando se presiona el botón o es la primera carga
if calculate_btn or 'first_load' not in st.session_state:
    st.session_state.first_load = True

    try:
        # Parse function
        func = sympify(func_str)
        f_num = lambdify((x, y), func, modules=['numpy'])

        # Create mesh
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_num(X, Y)
        Z = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)

        # Calcular derivadas parciales
        partial_x = diff(func, x)
        partial_y = diff(func, y)

        # Guardar todos los datos en session_state
        st.session_state.calculated_data = {
            'func': func,
            'func_str': func_str,
            'f_num': f_num,
            'X': X,
            'Y': Y,
            'Z': Z,
            'partial_x': partial_x,
            'partial_y': partial_y,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'point_x': point_x,
            'point_y': point_y
        }

    except Exception as e:
        st.error(f"Error al procesar la función: {str(e)}")
        st.info("Intente con una función más simple como: x**2 + y**2")
        st.session_state.calculated_data = None

# Mostrar las tabs solo si hay datos calculados
if st.session_state.calculated_data is not None:
    data = st.session_state.calculated_data
    func = data['func']
    func_str = data['func_str']
    f_num = data['f_num']
    X = data['X']
    Y = data['Y']
    Z = data['Z']
    partial_x = data['partial_x']
    partial_y = data['partial_y']
    stored_x_min = data['x_min']
    stored_x_max = data['x_max']
    stored_y_min = data['y_min']
    stored_y_max = data['y_max']
    stored_point_x = data['point_x']
    stored_point_y = data['point_y']

    # Tab 1: Visualización 3D
    with tab1:
        # Create 3D plot
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0, 'rgb(60, 120, 216)'],
                [0.25, 'rgb(100, 180, 255)'],
                [0.5, 'rgb(200, 240, 255)'],
                [0.75, 'rgb(255, 200, 150)'],
                [1, 'rgb(255, 100, 100)']
            ],
            opacity=0.95,
            showscale=True,
            colorbar=dict(title='z', thickness=20, len=0.7),
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="rgba(255,255,255,0.5)",
                    project=dict(z=True),
                    width=2
                )
            )
        )])

        # Add point
        if stored_x_min <= stored_point_x <= stored_x_max and stored_y_min <= stored_point_y <= stored_y_max:
            z_point = float(f_num(stored_point_x, stored_point_y))
            if not (np.isnan(z_point) or np.isinf(z_point)):
                fig.add_trace(go.Scatter3d(
                    x=[stored_point_x], y=[stored_point_y], z=[z_point],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='rgb(230, 30, 30)',
                        line=dict(color='white', width=3)
                    ),
                    name=f'P({stored_point_x}, {stored_point_y}, {z_point:.3f})'
                ))

        # Layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='x',
                    backgroundcolor='rgb(250, 250, 250)',
                    gridcolor='rgb(200, 200, 200)',
                    showbackground=True
                ),
                yaxis=dict(
                    title='y',
                    backgroundcolor='rgb(250, 250, 250)',
                    gridcolor='rgb(200, 200, 200)',
                    showbackground=True
                ),
                zaxis=dict(
                    title='z = f(x,y)',
                    backgroundcolor='rgb(250, 250, 250)',
                    gridcolor='rgb(200, 200, 200)',
                    showbackground=True
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                aspectmode='auto'
            ),
            title=f'f(x,y) = {func_str}',
            height=700
        )

        st.plotly_chart(fig, width="stretch")

        # Información básica
        st.subheader("Información de la función")
        z_min_val, z_max_val = np.nanmin(Z), np.nanmax(Z)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dominio", "ℝ²")
        with col2:
            st.metric("Rango aproximado", f"[{z_min_val:.4f}, {z_max_val:.4f}]")
        with col3:
            value_at_point = float(func.subs([(x, stored_point_x), (y, stored_point_y)]))
            st.metric(f"Valor en ({stored_point_x}, {stored_point_y})", f"{value_at_point:.6f}")

        # Aplicación práctica
        with st.expander("💡 Aplicación práctica (generada por IA)"):
            application = get_function_application(func_str)
            st.write(application)
        
    # Tab 2: Derivadas
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Derivadas Parciales")
            st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(partial_x))
            st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(partial_y))

        with col2:
            st.subheader(f"Gradiente en ({stored_point_x}, {stored_point_y})")
            try:
                grad_x = float(partial_x.subs([(x, stored_point_x), (y, stored_point_y)]))
                grad_y = float(partial_y.subs([(x, stored_point_x), (y, stored_point_y)]))
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)

                st.latex(r"\nabla f = (" + f"{grad_x:.6f}, {grad_y:.6f}" + ")")
                st.metric("Magnitud", f"{grad_mag:.6f}")
            except:
                st.error("No se pudo calcular el gradiente en este punto")
        
    # Tab 3: Optimización
    with tab3:
        st.subheader("Puntos Críticos")

        try:
            critical_points = solve([partial_x, partial_y], [x, y])

            if critical_points:
                if isinstance(critical_points, dict):
                    critical_points = [critical_points]

                for i, point in enumerate(critical_points):
                    if isinstance(point, dict):
                        px, py = float(point[x]), float(point[y])
                    else:
                        px, py = float(point[0]), float(point[1])

                    if stored_x_min <= px <= stored_x_max and stored_y_min <= py <= stored_y_max:
                        pz = float(func.subs([(x, px), (y, py)]))
                        st.success(f"**Punto {i+1}:** ({px:.6f}, {py:.6f}) → f = {pz:.6f}")
            else:
                st.info("No se encontraron puntos críticos en el dominio")
        except:
            st.warning("No se pudieron calcular los puntos críticos")

        st.subheader("Optimización con Restricción")
        constraint_str = st.text_input(
            "Restricción g(x,y) = 0 (opcional):",
            placeholder="x**2 + y**2 - 4"
        )

        if constraint_str and st.button("Optimizar con Lagrange"):
            try:
                constraint = sympify(constraint_str)
                g_num = lambdify((x, y), constraint, modules=['numpy'])

                def objective(vars):
                    return f_num(vars[0], vars[1])

                def constraint_func(vars):
                    return g_num(vars[0], vars[1])

                cons = {'type': 'eq', 'fun': constraint_func}
                x0 = [(stored_x_min + stored_x_max) / 2, (stored_y_min + stored_y_max) / 2]

                result = minimize(objective, x0, method='SLSQP', constraints=cons)

                if result.success:
                    st.success(f"**Punto óptimo:** ({result.x[0]:.6f}, {result.x[1]:.6f})")
                    st.metric("Valor de f", f"{result.fun:.6f}")
            except Exception as e:
                st.error(f"Error en la optimización: {str(e)}")
        
    # Tab 4: Integración
    with tab4:
        st.subheader("Integral Doble")

        col1, col2 = st.columns(2)
        with col1:
            int_x_min = st.number_input("x desde:", value=0.0)
            int_y_min = st.number_input("y desde:", value=0.0)
        with col2:
            int_x_max = st.number_input("x hasta:", value=1.0)
            int_y_max = st.number_input("y hasta:", value=1.0)

        if st.button("Calcular Integral"):
            try:
                # Intentar integración simbólica
                integral = integrate(integrate(func, (x, int_x_min, int_x_max)), (y, int_y_min, int_y_max))
                integral_value = float(integral)
                st.success(f"**Resultado:** {integral_value:.8f}")
            except:
                # Integración numérica
                from scipy import integrate as scipy_integrate

                def integrand(y_val, x_val):
                    return f_num(x_val, y_val)

                result, error = scipy_integrate.dblquad(
                    integrand, int_x_min, int_x_max, int_y_min, int_y_max
                )
                st.success(f"**Resultado (numérico):** {result:.8f}")
                st.info(f"Error estimado: {error:.2e}")

            st.markdown(f"""
            **Interpretación:** Volumen bajo la superficie en la región
            [{int_x_min}, {int_x_max}] × [{int_y_min}, {int_y_max}]
            """)

else:
    st.info("👆 Presiona el botón 'Calcular y Visualizar' en la barra lateral para comenzar.")
