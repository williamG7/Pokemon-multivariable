# Pokémon Multivariable - Predicción de Rendimiento OP Score

![Pikachu](https://i.pinimg.com/originals/d0/55/96/d0559662c49b354f89770b376c213e00.gif)

## Descripción

Proyecto de análisis de datos y machine learning que utiliza **regresión lineal multivariable** para predecir el **OP Score** (Overpowered Score) de Pokémon basándose en sus estadísticas base. El modelo analiza 800 Pokémon de diferentes generaciones para determinar qué características tienen mayor impacto en su rendimiento competitivo.

## Objetivo

Desarrollar un modelo predictivo capaz de estimar el rendimiento competitivo (OP Score) de cualquier Pokémon a partir de sus estadísticas base, identificando qué atributos son más determinantes en su efectividad en batalla.

### Variables del Modelo

**Variables Independientes (X):**
- **Total**: Suma base de estadísticas
- **HP**: Puntos de salud
- **Attack**: Capacidad de daño físico
- **Defense**: Resistencia a ataques físicos
- **Speed**: Agilidad y orden de turno
- **Generation**: Generación a la que pertenece el Pokémon

**Variable Dependiente (y):**
- **OP Score**: Métrica de rendimiento competitivo

## Dataset

- **Fuente**: [Pokemon OP Dataset](https://raw.githubusercontent.com/raimonizard/datasets/refs/heads/main/pokemon_op_wo_row_id.csv)
- **Tamaño**: 800 Pokémon
- **Características**: 14 columnas (estadísticas, tipos, generación, etc.)

![Dataset Preview](https://raw.githubusercontent.com/raimonizard/datasets/refs/heads/main/screenshots/pokemon_op_screenshot1.png)

## Tecnologías Utilizadas

```python
# Manipulación de datos
- pandas
- numpy

# Visualización
- matplotlib
- seaborn

# Machine Learning
- scikit-learn (LinearRegression, train_test_split, métricas)
```

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Carga y exploración del dataset
- Verificación de calidad de datos (valores nulos, tipos de datos)
- Análisis de correlaciones entre variables
- Visualización de matrices de correlación

### 2. Implementación de Regresión Lineal Multivariable
- Selección de variables independientes
- Entrenamiento del modelo con `LinearRegression().fit()`
- Generación de predicciones

### 3. Evaluación de Bondad de Ajuste
- Cálculo del coeficiente de determinación (R²)
- Métricas de error: MSE, RMSE, MAE
- Análisis de residuales

### 4. Análisis de Coeficientes
- Identificación del impacto de cada estadística
- Visualización de la importancia de las variables
- Interpretación de pesos en el modelo

### 5. Validación Predictiva
- Predicciones sobre el dataset completo
- Comparación de valores reales vs predichos
- Visualización de resultados

### 6. Simulación de Pokémon Sintéticos *(Extra)*
- Creación de Pokémon con estadísticas personalizadas
- Predicción de su OP Score teórico

### 7. División Train/Test *(Extra)*
- Separación 80/20 de los datos
- Evaluación de capacidad de generalización
- Validación cruzada

## Resultados Principales

### Rendimiento del Modelo
- **R² Score**: ~1.0000 (100% de explicación de la varianza)
- **Error promedio**: ±0.00 puntos
- **Conclusión**: Modelo con ajuste perfecto

### Variables Más Influyentes
1. **Total** (Coeficiente: 0.40) - Mayor impacto
2. **HP** (Coeficiente: 0.20)
3. **Defense** (Coeficiente: 0.15)
4. **Attack** (Coeficiente: 0.15)
5. **Speed** (Coeficiente: 0.05)
6. **Generation** (Coeficiente: 0.05)

### Correlación con OP Score
- **Total**: 0.9963 (correlación muy alta)
- **Attack**: 0.7666
- **HP**: 0.6519
- **Defense**: 0.6388
- **Speed**: 0.5441
- **Generation**: 0.0532 (correlación baja)

## Uso

### Instalación de Dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Ejecución del Notebook

Abre el archivo `Pokémon_multivariable_GuzmanWilliam.ipynb` en:
- **Google Colab** (recomendado): [Abrir en Colab](https://colab.research.google.com/github/williamG7/Pokemon-multivariable/blob/main/Pok%C3%A9mon_multivariable_GuzmanWilliam.ipynb)
- **Jupyter Notebook**: Ejecuta localmente con `jupyter notebook`

## Estructura del Proyecto

```
Pokemon-multivariable/
│
├── Pokémon_multivariable_GuzmanWilliam.ipynb  # Notebook principal
└── README.md                                   # Este archivo
```

## Conceptos de Machine Learning Aplicados

- **Regresión Lineal Multivariable**: Modelado de relaciones entre múltiples variables independientes y una dependiente
- **Coeficiente de Determinación (R²)**: Medida de bondad de ajuste del modelo
- **Análisis de Residuales**: Verificación de supuestos del modelo lineal
- **Train/Test Split**: Validación de capacidad de generalización
- **Feature Engineering**: Selección de variables predictoras

## Notas Técnicas

- El modelo utiliza `LinearRegression().fit()` para el entrenamiento
- Las predicciones se generan con `LinearRegression().predict(X)`
- El R² perfecto (1.0) sugiere que el OP Score podría estar calculado directamente como combinación lineal de las estadísticas
- Los valores nulos en "Type 2" (386) corresponden a Pokémon de un solo tipo

## Autor

**William Guzmán**

## Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---

⭐ Si este proyecto te fue útil, no olvides darle una estrella al repositorio!
