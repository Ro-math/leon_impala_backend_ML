
# Simulación León vs Impala - Aprendizaje por Refuerzo

Este proyecto implementa una simulación depredador-presa utilizando Aprendizaje por Refuerzo (Q-Learning) para entrenar a un agente León para cazar a un Impala.

## 1. Instalación y Configuración

### Requisitos
- Python 3.8+
- FastAPI
- Uvicorn
- NumPy

### Pasos de Instalación
1. Clonar el repositorio.
2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Ejecutar el Servidor
```bash
uvicorn app.main:app --reload
```
La API estará disponible en `http://localhost:8000`.

## 2. Base de Conocimientos

La Base de Conocimientos (KB) almacena las políticas aprendidas por el agente León.

### Representación
- **Q-Table**: Un diccionario que mapea `(PosiciónLeón, AcciónImpala, EstadoLeón)` a Valores-Q para cada `AcciónLeón`.
  - Formato de Clave: `x,y|accion_impala|estado_leon`
  - Valor: `{ "advance": 0.5, "attack": 0.8, ... }`
- **Abstracciones**: Una lista de reglas generalizadas derivadas de la Q-Table.
  - Ejemplo: `SI León en 1,9 Y León es normal Y Impala hace [look_left, look_right] ENTONCES advance`

### Almacenamiento
- El conocimiento se guarda como archivos JSON en `data/knowledge/`.
- `knowledge_final.json`: Se guarda al finalizar una sesión de entrenamiento.
- `knowledge_checkpoint.json`: Se guarda periódicamente durante el entrenamiento.

### Acceso y Actualización
- **Acceso**: El agente consulta la Q-Table para elegir la mejor acción (Explotación) o explora nuevas acciones (Exploración).
- **Actualización**: Los Valores-Q se actualizan después de cada paso usando la fórmula de Q-Learning:
  `Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))`

## 3. Proceso de Entrenamiento

El proceso de entrenamiento implica ejecutar múltiples "incursiones" (episodios de caza).

1. **Inicialización**: El León comienza en una posición válida aleatoria. El Impala comienza en (9,9).
2. **Ciclo de Pasos**:
   - El Agente observa el estado (Pos León, Acción Impala, Estado León).
   - El Agente elige una acción (Epsilon-Greedy).
   - El Entorno ejecuta el paso (Movimientos, lógica de huida).
   - Se calcula la recompensa (Positiva por captura, Negativa por escape/tiempo).
   - El Agente actualiza el Valor-Q.
3. **Terminación**: El episodio termina cuando el León atrapa al Impala o el Impala escapa.

### API de Entrenamiento
- `POST /api/training/start`: Iniciar una nueva sesión de entrenamiento.
- `POST /api/training/stop`: Detener el entrenamiento ordenadamente.
- `POST /api/training/resume`: Reanudar el entrenamiento.
- `GET /api/training/statistics`: Ver tasas de éxito y progreso.

## 4. Adquisición de Conocimiento y Abstracción

El sistema generaliza automáticamente el conocimiento para mejorar la eficiencia.

### Lógica de Abstracción
El `AbstractionEngine` escanea la Q-Table en busca de patrones:
1. Agrupa estados que difieren *solo* por la acción del Impala (ej. León en (1,9) frente a Impala Mirando Izquierda vs Mirando Derecha).
2. Si la mejor acción es la *misma* para estos estados (ej. "Avanzar"), crea una regla.
3. **Regla**: "Si León está en (1,9) y el Impala mira Izquierda O Derecha, ENTONCES Avanzar."

Esto permite al agente aplicar estrategias aprendidas a situaciones similares no vistas si se extiende para usar estas reglas en la toma de decisiones.

## 5. Ejemplos de Uso

### Iniciar Entrenamiento
```json
POST /api/training/start
{
  "num_incursions": 1000,
  "initial_positions": [1, 2, 3],
  "impala_mode": "random"
}
```

### Consultar Conocimiento
```http
GET /api/knowledge/base
```
Devuelve la Q-Table completa y las Abstracciones.

### Descargar Conocimiento
```http
GET /api/knowledge/download
```
Descarga el archivo `knowledge_base.json`.
