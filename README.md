# Clasificación de Resistencias

## Descripción

El objetivo del proyecto es desarrollar un algoritmo capaz de determinar el valor numérico de cada resistencia analizada. Para ello se procesan  40 imágenes de 10 resistencias eléctricas (4 por cada resistencia) en diferentes perspectivas, se aplica segmentación, transformaciones geométricas y detección de colores para las bandas de la resistencia.
Cada resistencia tiene 4 bandas de colores:
Las primeras 3 bandas definen el valor de la resistencia. 
La cuarta banda (siempre dorada) se debe ignorar, está ligeramente mas separada que las demas.
Se mostrará por consola el valor en Ohms de cada resistencia analizada. 

---

## Flujo del algoritmo

### 1️⃣ Transformación de imágenes
Cada imagen del conjunto de resistencias es cargada y convertida al espacio de color **HSV**.  
Luego se segmenta el fondo azul aplicando un rango de color, generando una **máscara binaria** que resalta la zona de interés.  

Se aplican filtros **gaussianos** y operaciones morfológicas de **clausura y dilatación** para eliminar imperfecciones y unificar regiones.  
A continuación, se detectan los **contornos** en la máscara y se selecciona el contorno más grande de forma rectangular, correspondiente al área de la resistencia.  

Si el rectángulo no es perfecto, se calcula el **rectángulo mínimo envolvente** y se reordenan los vértices (arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda) para aplicar una **transformación de perspectiva (homografía)**.  
De esta forma, se obtiene una imagen rectificada, con la resistencia orientada frontalmente y almacenada en la carpeta `Resistencias_out/`.

---

### 2️⃣ Detección de bandas de color
Del cuerpo rectificado de la resistencia se extrae automáticamente la región central mediante una máscara y una **proyección vertical** que limita los márgenes a la zona útil.  

Para detectar las bandas, se calcula el **gradiente vertical (Sobel)** y se proyecta sobre el eje X, identificando los **picos máximos** que corresponden a los bordes de las bandas de color.  
Luego, para cada banda detectada, se toma una **franja vertical** alrededor de su centro y se calcula el **color dominante** comparando el promedio HSV con los rangos predefinidos de los colores estándar:  
negro, marrón, rojo, naranja, amarillo, verde, azul, violeta, gris, blanco, plata y dorado.

El sistema devuelve la secuencia de colores identificados para cada resistencia procesada.

---

### 3️⃣ Cálculo del valor de la resistencia
Se interpreta la secuencia de colores detectada según el **código de colores de resistencias**.  
- Los **dos primeros colores** representan los dígitos significativos.  
- El **tercer color** indica el multiplicador.  
- La **cuarta banda (dorado o plateado)** representa la tolerancia, pero no se incluye en el valor nominal.  

El valor calculado se formatea automáticamente en:
- **MΩ** si es ≥ 1.000.000 Ω  
- **kΩ** si es ≥ 1.000 Ω  
- **Ω** en caso contrario.  

Finalmente, el programa imprime el resultado por consola, mostrando el **valor estimado en ohmios** y la **combinación de colores detectados** para cada imagen.

---
## Estrcutura del repositorio

```
PDI-TP2-Clasificacion-Resistencias/
│
├── imagenes/ # Carpeta con las imágenes de resistencias
│ ├── R1_a.jpg
│ ├── R2_a.jpg
│ ├── ...
│
├── clasificacion_resistencias.py # Script principal del proyecto
├── requirements.txt # Dependencias necesarias
└── README.md # Documentación del proyecto
```
---


## Ejecución del proyecto con entorno virtual

Desde la carpeta raíz del repositorio:

```bash
# 1) Crear entorno virtual
python -m venv venv

# 2) Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux / macOS:
source venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Ejecutar el script principal
python src/clasificacion_resistencias.py

# 5) (Opcional) Desactivar el entorno virtual al finalizar
deactivate
