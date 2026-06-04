from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp
from sympy import exp, log, ln, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, E, pi
import matplotlib.pyplot as plt
import io
import base64

import matplotlib
matplotlib.use('Agg')  # IMPORTANTE: Para Render que no tiene GUI

app = Flask(__name__)

class OptimizacionWolfe:
    def __init__(self, metodo, funcion_str, x0, max_iter, tol, c1, c2):
        self.metodo = metodo
        self.funcion_str = funcion_str
        self.x0 = np.array(x0)
        self.max_iter = max_iter
        self.tol = tol
        self.c1 = c1  # Primera condición de Wolfe (Armijo)
        self.c2 = c2  # Segunda condición de Wolfe (curvatura)
        
        # Variables simbólicas
        self.n = len(x0)
        self.x = sp.symbols([f'x{i}' for i in range(self.n)])
        
        # Diccionario de funciones matemáticas disponibles
        math_functions = {
            'exp': exp, 'log': log, 'ln': ln, 'sin': sin, 'cos': cos, 'tan': tan,
            'asin': asin, 'acos': acos, 'atan': atan, 'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'E': E, 'pi': pi
        }
        self.f = sp.sympify(funcion_str, locals=math_functions)
        
        # Gradiente y Hessiano
        self.grad = [sp.diff(self.f, var) for var in self.x]
        self.hess = [[sp.diff(g, var) for var in self.x] for g in self.grad]
        
        self.historial_error = []
        
    def evaluar_funcion(self, x_val):
        subs = {self.x[i]: x_val[i] for i in range(self.n)}
        return float(self.f.subs(subs))
    
    def evaluar_gradiente(self, x_val):
        subs = {self.x[i]: x_val[i] for i in range(self.n)}
        return np.array([float(g.subs(subs)) for g in self.grad])
    
    def evaluar_hessiano(self, x_val):
        subs = {self.x[i]: x_val[i] for i in range(self.n)}
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                H[i,j] = float(self.hess[i][j].subs(subs))
        return H
    
    def wolfe_conditions(self, x, p, alpha):
        """Verifica condiciones de Wolfe para step length alpha"""
        x_new = x + alpha * p
        f_x = self.evaluar_funcion(x)
        f_x_new = self.evaluar_funcion(x_new)
        grad_x = self.evaluar_gradiente(x)
        
        # Condición 1: Suficiente decrecimiento (Armijo)
        cond1 = f_x_new <= f_x + self.c1 * alpha * np.dot(grad_x, p)
        
        # Condición 2: Curvatura
        grad_new = self.evaluar_gradiente(x_new)
        cond2 = np.dot(grad_new, p) >= self.c2 * np.dot(grad_x, p)
        
        return cond1 and cond2
    
    def line_search_wolfe(self, x, p):
        """Búsqueda de línea con condiciones de Wolfe"""
        alpha = 1.0
        rho = 0.5  # Factor de reducción
        max_iter_ls = 100
        
        for _ in range(max_iter_ls):
            if self.wolfe_conditions(x, p, alpha):
                return alpha
            alpha *= rho
        
        return alpha  # Retorna último alpha si no encuentra
    
    def metodo_gradiente(self):
        x = self.x0.copy()
        historial = [x.copy()]
        
        for k in range(self.max_iter):
            grad = self.evaluar_gradiente(x)
            error = np.linalg.norm(grad)
            self.historial_error.append(error)
            
            if error < self.tol:
                break
            
            # Dirección: negativo del gradiente
            p = -grad
            
            # Búsqueda de línea con Wolfe
            alpha = self.line_search_wolfe(x, p)
            
            # Actualización
            x = x + alpha * p
            historial.append(x.copy())
        
        return x, k+1, historial
    
    def metodo_gradiente_conjugado(self):
        x = self.x0.copy()
        historial = [x.copy()]
        grad_prev = None
        p_prev = None
        
        for k in range(self.max_iter):
            grad = self.evaluar_gradiente(x)
            error = np.linalg.norm(grad)
            self.historial_error.append(error)
            
            if error < self.tol:
                break
            
            # Calcular dirección conjugada
            if k == 0:
                p = -grad
            else:
                # Método de Fletcher-Reeves
                beta = np.dot(grad, grad) / np.dot(grad_prev, grad_prev)
                p = -grad + beta * p_prev
            
            # Búsqueda de línea con Wolfe
            alpha = self.line_search_wolfe(x, p)
            
            # Actualización
            x_new = x + alpha * p
            x = x_new
            grad_prev = grad
            p_prev = p
            historial.append(x.copy())
        
        return x, k+1, historial
    
    def metodo_newton(self):
        x = self.x0.copy()
        historial = [x.copy()]
        
        for k in range(self.max_iter):
            grad = self.evaluar_gradiente(x)
            error = np.linalg.norm(grad)
            self.historial_error.append(error)
            
            if error < self.tol:
                break
            
            hess = self.evaluar_hessiano(x)
            
            try:
                # Dirección de Newton
                p = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Si Hessiano es singular, usar gradiente
                p = -grad
            
            # Búsqueda de línea con Wolfe
            alpha = self.line_search_wolfe(x, p)
            
            # Actualización
            x = x + alpha * p
            historial.append(x.copy())
        
        return x, k+1, historial
    
    def ejecutar(self):
        if self.metodo == 'gradiente':
            return self.metodo_gradiente()
        elif self.metodo == 'gradiente_conjugado':
            return self.metodo_gradiente_conjugado()
        elif self.metodo == 'newton':
            return self.metodo_newton()
        else:
            raise ValueError("Método no soportado")
    
    def generar_grafico(self):
        """Genera gráfico de convergencia"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iteraciones = list(range(1, len(self.historial_error) + 1))
        ax.semilogy(iteraciones, self.historial_error, 'b-', marker='o', markersize=4)
        ax.set_xlabel('Iteraciones', fontsize=12)
        ax.set_ylabel('Error (norma del gradiente)', fontsize=12)
        ax.set_title('Convergencia del método de optimización', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Guardar en base64 para enviar al frontend
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimizar', methods=['POST'])
def optimizar():
    try:
        data = request.json
        
        metodo = data['metodo']
        funcion = data['funcion']
        x0 = data['x0']
        max_iter = int(data['max_iter'])
        tol = float(data['tol'])
        c1 = float(data['c1'])
        c2 = float(data['c2'])
        
        # Validar parámetros de Wolfe
        if not (0 < c1 < c2 < 1):
            return jsonify({'error': 'c1 y c2 deben satisfacer 0 < c1 < c2 < 1'}), 400
        
        # Crear y ejecutar optimizador
        opt = OptimizacionWolfe(metodo, funcion, x0, max_iter, tol, c1, c2)
        x_min, n_iter, historial = opt.ejecutar()
        
        # Valor final de la función
        f_min = opt.evaluar_funcion(x_min)
        error_final = opt.historial_error[-1] if opt.historial_error else 0
        
        # Generar gráfico
        grafico = opt.generar_grafico()
        
        resultado = {
            'exito': True,
            'punto_minimo': x_min.tolist(),
            'valor_funcion': f_min,
            'iteraciones': n_iter,
            'error_final': error_final,
            'criterio_parada': f'Norma del gradiente < {tol}',
            'grafico': grafico,
            'historial_puntos': [p.tolist() for p in historial]
        }
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({'exito': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
