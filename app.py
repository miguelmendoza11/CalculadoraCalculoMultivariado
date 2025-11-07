from flask import Flask, render_template_string, request, jsonify
import numpy as np
import sympy as sp
from sympy import symbols, lambdify, sympify, diff, integrate, solve
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import json
import requests
import os

app = Flask(__name__)

# Configuración de Google Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDjcmpdzo4HiOwi7Ct-NF5m2PUqe0uWTFc')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calculadora de Cálculo Multivariable</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 0;
        }
        
        .sidebar {
            background: #f8f9fa;
            padding: 30px;
            border-right: 1px solid #dee2e6;
        }
        
        .content-area {
            padding: 30px;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section-title {
            color: #667eea;
            font-size: 1.3em;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #495057;
            font-weight: 500;
        }
        
        input[type="text"], select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            margin-top: 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab {
            padding: 12px 25px;
            background: transparent;
            border: none;
            cursor: pointer;
            font-size: 15px;
            font-weight: 500;
            color: #6c757d;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab:hover {
            color: #667eea;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .result-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid #667eea;
        }
        
        .result-title {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .result-content {
            color: #495057;
            line-height: 1.6;
        }
        
        #plot {
            width: 100%;
            height: 700px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            background: white;
            border: 1px solid #e0e0e0;
        }
        
        .help-text {
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 5px;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .example-functions {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .example-title {
            font-weight: 600;
            color: #0066cc;
            margin-bottom: 10px;
        }
        
        .example-item {
            padding: 8px;
            cursor: pointer;
            border-radius: 5px;
            transition: background 0.2s;
            font-size: 0.9em;
        }
        
        .example-item:hover {
            background: #cce5ff;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Calculadora de Cálculo Multivariable</h1>
            <p class="subtitle">Visualización, Cálculo e Interpretación de Funciones Multivariables</p>
        </header>
        
        <div class="main-content">
            <div class="sidebar">
                <div class="section">
                    <div class="section-title">Función</div>
                    <div class="input-group">
                        <label>Ingrese la función f(x,y):</label>
                        <input type="text" id="function" placeholder="x**2 + y**2" value="x**2 + y**2">
                        <div class="help-text">Ejemplo: x**2 + y**2, sin(x)*cos(y), exp(-(x**2 + y**2))</div>
                    </div>
                    
                    <div class="example-functions">
                        <div class="example-title">Funciones de ejemplo:</div>
                        <div class="example-item" onclick="setFunction('x**2 + y**2')">• Paraboloide: x² + y²</div>
                        <div class="example-item" onclick="setFunction('sin(x)*cos(y)')">• Seno-Coseno: sin(x)cos(y)</div>
                        <div class="example-item" onclick="setFunction('x**2 - y**2')">• Silla de montar: x² - y²</div>
                        <div class="example-item" onclick="setFunction('exp(-(x**2 + y**2))')">• Gaussiana: e^(-(x²+y²))</div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">Rango de visualización</div>
                    <div class="grid-2">
                        <div class="input-group">
                            <label>x min:</label>
                            <input type="text" id="x_min" value="-5" placeholder="-5">
                        </div>
                        <div class="input-group">
                            <label>x max:</label>
                            <input type="text" id="x_max" value="5" placeholder="5">
                        </div>
                        <div class="input-group">
                            <label>y min:</label>
                            <input type="text" id="y_min" value="-5" placeholder="-5">
                        </div>
                        <div class="input-group">
                            <label>y max:</label>
                            <input type="text" id="y_max" value="5" placeholder="5">
                        </div>
                    </div>
                    <div class="help-text">Dejar vacío para usar valores por defecto</div>
                </div>
                
                <div class="section">
                    <div class="section-title">Punto de evaluación</div>
                    <div class="grid-2">
                        <div class="input-group">
                            <label>x₀:</label>
                            <input type="text" id="point_x" value="1" placeholder="1">
                        </div>
                        <div class="input-group">
                            <label>y₀:</label>
                            <input type="text" id="point_y" value="1" placeholder="1">
                        </div>
                    </div>
                </div>
                
                <button class="btn" onclick="calculate()">Calcular y Visualizar</button>
            </div>
            
            <div class="content-area">
                <div class="tabs">
                    <button class="tab active" onclick="showTab('visualization')">Visualización 3D</button>
                    <button class="tab" onclick="showTab('derivatives')">Derivadas</button>
                    <button class="tab" onclick="showTab('optimization')">Optimización</button>
                    <button class="tab" onclick="showTab('integration')">Integración</button>
                </div>
                
                <div id="visualization" class="tab-content active">
                    <div id="plot"></div>
                    <div id="basic-results"></div>
                </div>
                
                <div id="derivatives" class="tab-content">
                    <div id="derivatives-results"></div>
                </div>
                
                <div id="optimization" class="tab-content">
                    <div class="input-group">
                        <label>Restricción g(x,y) = 0 (opcional):</label>
                        <input type="text" id="constraint" placeholder="x**2 + y**2 - 4">
                        <div class="help-text">Ejemplo: x**2 + y**2 - 4 (círculo de radio 2)</div>
                    </div>
                    <button class="btn" onclick="optimize()">Optimizar</button>
                    <div id="optimization-results"></div>
                </div>
                
                <div id="integration" class="tab-content">
                    <div class="section-title">Límites de integración</div>
                    <div class="grid-2">
                        <div class="input-group">
                            <label>x desde:</label>
                            <input type="text" id="int_x_min" value="0">
                        </div>
                        <div class="input-group">
                            <label>x hasta:</label>
                            <input type="text" id="int_x_max" value="1">
                        </div>
                        <div class="input-group">
                            <label>y desde:</label>
                            <input type="text" id="int_y_min" value="0">
                        </div>
                        <div class="input-group">
                            <label>y hasta:</label>
                            <input type="text" id="int_y_max" value="1">
                        </div>
                    </div>
                    <button class="btn" onclick="integrateFunction()">Integrar</button>
                    <div id="integration-results"></div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Calculando...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function setFunction(func) {
            document.getElementById('function').value = func;
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        async function calculate() {
            showLoading(true);
            
            const data = {
                function: document.getElementById('function').value,
                x_min: document.getElementById('x_min').value || '-5',
                x_max: document.getElementById('x_max').value || '5',
                y_min: document.getElementById('y_min').value || '-5',
                y_max: document.getElementById('y_max').value || '5',
                point_x: document.getElementById('point_x').value || '1',
                point_y: document.getElementById('point_y').value || '1'
            };
            
            // Convertir a números
            data.x_min = parseFloat(data.x_min);
            data.x_max = parseFloat(data.x_max);
            data.y_min = parseFloat(data.y_min);
            data.y_max = parseFloat(data.y_max);
            data.point_x = parseFloat(data.point_x);
            data.point_y = parseFloat(data.point_y);
            
            // Validar que sean números válidos
            if (isNaN(data.x_min)) data.x_min = -5;
            if (isNaN(data.x_max)) data.x_max = 5;
            if (isNaN(data.y_min)) data.y_min = -5;
            if (isNaN(data.y_max)) data.y_max = 5;
            if (isNaN(data.point_x)) data.point_x = 1;
            if (isNaN(data.point_y)) data.point_y = 1;
            
            try {
                const response = await fetch('/calculate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.error) {
                    alert('Error: ' + result.error);
                    showLoading(false);
                    return;
                }
                
                // Plot 3D surface with GeoGebra style
                Plotly.newPlot('plot', JSON.parse(result.plot_data), {
                    margin: {l: 0, r: 0, t: 40, b: 0},
                    showlegend: true,
                    legend: {
                        x: 0.02,
                        y: 0.98,
                        bgcolor: 'rgba(255, 255, 255, 0.8)',
                        bordercolor: 'rgb(200, 200, 200)',
                        borderwidth: 1
                    }
                }, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['toImage'],
                    displaylogo: false
                });
                
                // Show basic results
                document.getElementById('basic-results').innerHTML = `
                    <div class="result-box">
                        <div class="result-title">Información de la función</div>
                        <div class="result-content">
                            <p><strong>Dominio:</strong> ${result.domain}</p>
                            <p><strong>Rango aproximado:</strong> ${result.range}</p>
                            <p><strong>Valor en (${data.point_x}, ${data.point_y}):</strong> ${result.value_at_point}</p>
                            <p><strong>Aplicación práctica:</strong> ${result.application}</p>
                        </div>
                    </div>
                `;
                
                // Show derivatives
                document.getElementById('derivatives-results').innerHTML = `
                    <div class="result-box">
                        <div class="result-title">Derivadas Parciales</div>
                        <div class="result-content">
                            <p><strong>∂f/∂x =</strong> ${result.partial_x}</p>
                            <p><strong>∂f/∂y =</strong> ${result.partial_y}</p>
                        </div>
                    </div>
                    <div class="result-box">
                        <div class="result-title">Gradiente en (${data.point_x}, ${data.point_y})</div>
                        <div class="result-content">
                            <p><strong>∇f =</strong> (${result.gradient[0]}, ${result.gradient[1]})</p>
                            <p><strong>Magnitud:</strong> ${result.gradient_magnitude}</p>
                        </div>
                    </div>
                `;
                
                showLoading(false);
            } catch (error) {
                alert('Error en el cálculo: ' + error);
                showLoading(false);
            }
        }
        
        async function optimize() {
            showLoading(true);
            
            const data = {
                function: document.getElementById('function').value,
                constraint: document.getElementById('constraint').value,
                x_min: parseFloat(document.getElementById('x_min').value),
                x_max: parseFloat(document.getElementById('x_max').value),
                y_min: parseFloat(document.getElementById('y_min').value),
                y_max: parseFloat(document.getElementById('y_max').value)
            };
            
            try {
                const response = await fetch('/optimize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.error) {
                    alert('Error: ' + result.error);
                    showLoading(false);
                    return;
                }
                
                let html = '<div class="result-box"><div class="result-title">Puntos Críticos</div><div class="result-content">';
                
                if (result.critical_points && result.critical_points.length > 0) {
                    result.critical_points.forEach((point, i) => {
                        html += `<p><strong>Punto ${i+1}:</strong> (${point[0]}, ${point[1]}) → f = ${point[2]}</p>`;
                    });
                } else {
                    html += '<p>No se encontraron puntos críticos en el dominio.</p>';
                }
                
                html += '</div></div>';
                
                if (result.lagrange) {
                    html += `
                        <div class="result-box">
                            <div class="result-title">Optimización con Restricción (Lagrange)</div>
                            <div class="result-content">
                                <p><strong>Punto óptimo:</strong> (${result.lagrange.point[0]}, ${result.lagrange.point[1]})</p>
                                <p><strong>Valor de f:</strong> ${result.lagrange.value}</p>
                                <p><strong>Multiplicador λ:</strong> ${result.lagrange.lambda || 'N/A'}</p>
                            </div>
                        </div>
                    `;
                }
                
                document.getElementById('optimization-results').innerHTML = html;
                showLoading(false);
            } catch (error) {
                alert('Error en la optimización: ' + error);
                showLoading(false);
            }
        }
        
        async function integrateFunction() {
            showLoading(true);
            
            const data = {
                function: document.getElementById('function').value,
                x_min: parseFloat(document.getElementById('int_x_min').value),
                x_max: parseFloat(document.getElementById('int_x_max').value),
                y_min: parseFloat(document.getElementById('int_y_min').value),
                y_max: parseFloat(document.getElementById('int_y_max').value)
            };
            
            try {
                const response = await fetch('/integrate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.error) {
                    alert('Error: ' + result.error);
                    showLoading(false);
                    return;
                }
                
                document.getElementById('integration-results').innerHTML = `
                    <div class="result-box">
                        <div class="result-title">Integral Doble</div>
                        <div class="result-content">
                            <p><strong>Resultado:</strong> ${result.integral_value}</p>
                            <p><strong>Interpretación:</strong> Volumen bajo la superficie en la región especificada</p>
                            <p><strong>Región:</strong> [${data.x_min}, ${data.x_max}] × [${data.y_min}, ${data.y_max}]</p>
                        </div>
                    </div>
                `;
                
                showLoading(false);
            } catch (error) {
                alert('Error en la integración: ' + error);
                showLoading(false);
            }
        }
        
        // Initial calculation
        window.onload = function() {
            calculate();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def get_function_application(function_str):
    """Obtiene una aplicación práctica de la función usando Gemini API"""
    try:
        if GEMINI_API_KEY == 'TU_API_KEY_AQUI' or not GEMINI_API_KEY:
            return "Configure su API key de Gemini para ver aplicaciones prácticas."
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        prompt = f"""Describe brevemente (máximo 2-3 oraciones) una aplicación práctica real de la función matemática f(x,y) = {function_str} en ingeniería, física, economía o ciencias. 
        Sé específico y conciso. No uses formato markdown ni asteriscos."""
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")  # Debug
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")  # Debug
            
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0]:
                    if 'parts' in result['candidates'][0]['content']:
                        text = result['candidates'][0]['content']['parts'][0]['text']
                        return text.strip()
        else:
            print(f"Error response: {response.text}")  # Debug
        
        return "Esta función tiene múltiples aplicaciones en modelado matemático y análisis científico."
        
    except Exception as e:
        print(f"Exception en Gemini API: {str(e)}")  # Debug
        return "Esta función tiene aplicaciones en diversos campos de la ciencia e ingeniería."

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        func_str = data['function']
        
        # Usar valores por defecto si están vacíos
        x_min = data.get('x_min', -5)
        x_max = data.get('x_max', 5)
        y_min = data.get('y_min', -5)
        y_max = data.get('y_max', 5)
        point_x = data.get('point_x', 1)
        point_y = data.get('point_y', 1)
        
        # Convertir a float y manejar valores vacíos
        try:
            x_min = float(x_min) if x_min != '' and x_min is not None else -5
        except:
            x_min = -5
            
        try:
            x_max = float(x_max) if x_max != '' and x_max is not None else 5
        except:
            x_max = 5
            
        try:
            y_min = float(y_min) if y_min != '' and y_min is not None else -5
        except:
            y_min = -5
            
        try:
            y_max = float(y_max) if y_max != '' and y_max is not None else 5
        except:
            y_max = 5
            
        try:
            point_x = float(point_x) if point_x != '' and point_x is not None else 1
        except:
            point_x = 1
            
        try:
            point_y = float(point_y) if point_y != '' and point_y is not None else 1
        except:
            point_y = 1
        
        # Define symbols
        x, y = symbols('x y')
        
        # Parse function
        func = sympify(func_str)
        
        # Create numerical function
        f_num = lambdify((x, y), func, modules=['numpy'])
        
        # Create high-resolution mesh for plotting (100x100 para superficies suaves)
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Evaluate function with error handling
        try:
            Z = f_num(X, Y)
            # Replace inf and nan values
            Z = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)
        except:
            Z = X**2 + Y**2  # Fallback to simple function
        
        # Create 3D surface plot with GeoGebra style
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0, 'rgb(60, 120, 216)'],      # Azul GeoGebra
                [0.25, 'rgb(100, 180, 255)'],  # Azul claro
                [0.5, 'rgb(200, 240, 255)'],   # Celeste
                [0.75, 'rgb(255, 200, 150)'],  # Naranja claro
                [1, 'rgb(255, 100, 100)']      # Rojo-naranja
            ],
            opacity=0.95,
            showscale=True,
            colorbar=dict(
                title='z',
                thickness=20,
                len=0.7
            ),
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(
                x=1000,
                y=1000,
                z=1000
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="rgba(255,255,255,0.5)",
                    project=dict(z=True),
                    width=2
                )
            ),
            hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
        )])
        
        # Add point if within bounds
        if x_min <= point_x <= x_max and y_min <= point_y <= y_max:
            try:
                z_point = float(f_num(point_x, point_y))
                if np.isnan(z_point) or np.isinf(z_point):
                    z_point = 0.0
            except:
                z_point = 0.0
                
            fig.add_trace(go.Scatter3d(
                x=[point_x], y=[point_y], z=[z_point],
                mode='markers',
                marker=dict(
                    size=10,
                    color='rgb(230, 30, 30)',  # Rojo GeoGebra
                    line=dict(color='white', width=3),
                    symbol='circle'
                ),
                name=f'P({point_x}, {point_y}, {z_point:.3f})',
                showlegend=True,
                hovertemplate=f'<b>Punto evaluado</b><br>x: {point_x}<br>y: {point_y}<br>z: {z_point:.6f}<extra></extra>'
            ))
        
        # GeoGebra style layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='x',
                    backgroundcolor='rgb(250, 250, 250)',
                    gridcolor='rgb(200, 200, 200)',
                    showbackground=True,
                    zerolinecolor='rgb(100, 100, 100)',
                    gridwidth=2
                ),
                yaxis=dict(
                    title='y',
                    backgroundcolor='rgb(250, 250, 250)',
                    gridcolor='rgb(200, 200, 200)',
                    showbackground=True,
                    zerolinecolor='rgb(100, 100, 100)',
                    gridwidth=2
                ),
                zaxis=dict(
                    title='z = f(x,y)',
                    backgroundcolor='rgb(250, 250, 250)',
                    gridcolor='rgb(200, 200, 200)',
                    showbackground=True,
                    zerolinecolor='rgb(100, 100, 100)',
                    gridwidth=2
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                aspectmode='auto'
            ),
            title=dict(
                text=f'f(x,y) = {func_str}',
                font=dict(size=16, color='rgb(50, 50, 50)')
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=12, color='rgb(80, 80, 80)')
        )
        
        plot_data = fig.to_json()
        
        # Calculate partial derivatives
        partial_x = diff(func, x)
        partial_y = diff(func, y)
        
        # Evaluate at point
        try:
            value_at_point = float(func.subs([(x, point_x), (y, point_y)]))
            if np.isnan(value_at_point) or np.isinf(value_at_point):
                value_at_point = 0.0
        except:
            value_at_point = 0.0
            
        try:
            grad_x = float(partial_x.subs([(x, point_x), (y, point_y)]))
            grad_y = float(partial_y.subs([(x, point_y), (y, point_y)]))
            if np.isnan(grad_x) or np.isinf(grad_x):
                grad_x = 0.0
            if np.isnan(grad_y) or np.isinf(grad_y):
                grad_y = 0.0
        except:
            grad_x = 0.0
            grad_y = 0.0
        
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Determine domain and range
        domain = f"ℝ² (todos los puntos (x,y) donde la función está definida)"
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        range_str = f"[{z_min:.4f}, {z_max:.4f}]"
        
        # Get practical application from Gemini
        application = get_function_application(func_str)
        
        return jsonify({
            'plot_data': plot_data,
            'domain': domain,
            'range': range_str,
            'value_at_point': f"{value_at_point:.6f}",
            'partial_x': str(partial_x),
            'partial_y': str(partial_y),
            'gradient': [f"{grad_x:.6f}", f"{grad_y:.6f}"],
            'gradient_magnitude': f"{gradient_mag:.6f}",
            'application': application
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/optimize', methods=['POST'])
def optimize_function():
    try:
        data = request.json
        func_str = data['function']
        constraint_str = data.get('constraint', '')
        x_min, x_max = data['x_min'], data['x_max']
        y_min, y_max = data['y_min'], data['y_max']
        
        x, y = symbols('x y')
        func = sympify(func_str)
        
        # Find critical points
        partial_x = diff(func, x)
        partial_y = diff(func, y)
        
        critical_points = solve([partial_x, partial_y], [x, y])
        
        # Evaluate critical points
        result_points = []
        if isinstance(critical_points, list):
            for point in critical_points:
                if len(point) == 2:
                    px, py = float(point[0]), float(point[1])
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        pz = float(func.subs([(x, px), (y, py)]))
                        result_points.append([f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"])
        elif isinstance(critical_points, dict):
            px, py = float(critical_points[x]), float(critical_points[y])
            if x_min <= px <= x_max and y_min <= py <= y_max:
                pz = float(func.subs([(x, px), (y, py)]))
                result_points.append([f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"])
        
        response = {'critical_points': result_points}
        
        # Lagrange multipliers if constraint provided
        if constraint_str:
            constraint = sympify(constraint_str)
            f_num = lambdify((x, y), func, modules=['numpy'])
            g_num = lambdify((x, y), constraint, modules=['numpy'])
            
            # Optimization with constraint
            def objective(vars):
                return f_num(vars[0], vars[1])
            
            def constraint_func(vars):
                return g_num(vars[0], vars[1])
            
            cons = {'type': 'eq', 'fun': constraint_func}
            x0 = [(x_min + x_max) / 2, (y_min + y_max) / 2]
            
            result = minimize(objective, x0, method='SLSQP', constraints=cons)
            
            if result.success:
                response['lagrange'] = {
                    'point': [f"{result.x[0]:.6f}", f"{result.x[1]:.6f}"],
                    'value': f"{result.fun:.6f}"
                }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/integrate', methods=['POST'])
def integrate_function():
    try:
        data = request.json
        func_str = data['function']
        x_min, x_max = data['x_min'], data['x_max']
        y_min, y_max = data['y_min'], data['y_max']
        
        x, y = symbols('x y')
        func = sympify(func_str)
        
        # Double integration
        try:
            integral = integrate(integrate(func, (x, x_min, x_max)), (y, y_min, y_max))
            integral_value = float(integral)
            result_str = f"{integral_value:.8f}"
        except:
            # Numerical integration
            f_num = lambdify((x, y), func, modules=['numpy'])
            from scipy import integrate as scipy_integrate
            
            def integrand(y_val, x_val):
                return f_num(x_val, y_val)
            
            result, error = scipy_integrate.dblquad(
                integrand, x_min, x_max, y_min, y_max
            )
            result_str = f"{result:.8f} (numérico)"
        
        return jsonify({'integral_value': result_str})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)