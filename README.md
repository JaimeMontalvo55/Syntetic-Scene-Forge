# Synthetic Scene Forge

Aplicación local independiente para generar datasets sintéticos de detección de
objetos a partir de modelos STL. El núcleo de renderizado reutiliza el flujo
original de transformación de mallas, renderizado RGB/segmentación, cajas
delimitadoras, separación mínima entre objetos, profundidad y división
`train`/`val`.

El proyecto no contiene ROS, nodos de percepción ni detectores.

## Instalación

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux puede ser necesario instalar las bibliotecas OpenGL del sistema para
`pyrender` en modo offscreen.

En Windows la aplicación crea el contexto usando una configuración compatible
con controladores que no exportan `WGL_ARB_pixel_format`. Si el controlador no
ofrece OpenGL 3.3, actualízalo antes de generar el dataset.

El proyecto fija `pyglet<2` porque es la versión compatible con `pyrender 0.1.x`.

## Ejecución

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La aplicación permite cargar cualquier número de STL, asignar nombre/clase,
seleccionar colores aleatorios, cargar imágenes y vídeos, configurar los
parámetros de generación y generar `images`, `labels`, `depth`, `previews`,
`reports` y `data.yaml`. La distribución de objetos por imagen se adapta al
número de modelos seleccionados y usa colores de segmentación distintos para
cada clase.
También se exportan máscaras 2D (`masks/`), anotaciones COCO por partición
(`annotations_train.json`, `annotations_val.json` y `annotations_test.json`) y
un informe estadístico en `reports/generation_report.json`. Las imágenes sin
objetos pueden configurarse mediante `negative_ratio`.
La generación se ejecuta en segundo plano y ofrece un botón para detenerla.
Al cancelar, se eliminan los archivos creados durante esa ejecución, sin borrar
los archivos que ya existían en la carpeta de salida.
El modo combinado alterna entre colores aleatorios y escenarios de imágenes o
vídeos.

La posición de profundidad se configura en metros: la distancia media es el
valor central de la distribución triangular y las distancias mínima y máxima
definen sus límites. Los STL se normalizan automáticamente para que su mayor
dimensión mida 1 metro; así la escala del archivo no altera la interpretación
de las distancias.
La interfaz también permite variar la iluminación visual del material de cada
objeto mediante un mínimo y un máximo entre 0 y 1. El valor se elige
aleatoriamente por objeto y escena para evitar que todos destaquen igual sobre
el fondo. No se aplica emisión: los objetos no actúan como fuentes de luz.

Si todas las escenas se rechazan por etiquetas, la interfaz muestra los
píxeles visibles detectados por clase. Un valor `0px` indica que el STL no
está llegando a la cámara o no se está rasterizando; valores menores de
`20px` indican que el objeto queda demasiado pequeño para el umbral de
visibilidad.

Los STL no contienen unidades. Los modelos se normalizan para trabajar en una
escala compatible con la cámara. La validación solo detiene modelos claramente
fuera de escala; modelos grandes pero todavía visibles se permiten y se
comprueban mediante la máscara de segmentación.

Cada STL recibe una orientación aleatoria independiente. La interfaz muestra
los ejes locales del modelo con la convención X (izquierda/derecha), Y
(abajo/arriba) y Z (profundidad). Además de seleccionar uno o varios ejes y
definir el paso angular, se puede establecer una orientación inicial fija en
grados para cada eje. Esta corrección se aplica antes de la rotación aleatoria,
por lo que permite colocar correctamente modelos exportados con una orientación
distinta. Los ángulos aleatorios se eligen entre 0 y 360 grados usando el paso;
un eje no seleccionado no rota. La configuración exportada conserva ambos
conjuntos de ángulos para reproducir la generación.
La interfaz incluye un visualizador por STL con los ejes locales superpuestos:
X rojo, Y verde y Z azul.

El flujo original no incluye extracción de fotogramas de vídeo ni
procedimientos de escenarios adicionales. Por ello, vídeo e imagen se ofrecen
como fuentes de escenarios mediante adaptadores independientes.

## Estructura

```text
app.py
src/synthetic_dataset_generator/
  config.py       Parámetros y configuración serializable
  models.py       Carga y normalización de STL
  sources.py      Fondos, imágenes y extracción de fotogramas
  generation.py   Renderizado, composición y etiquetado
  export.py       Exportación de etiquetas y configuración
example_config.json
```

Los datasets generados y los ficheros subidos están excluidos por `.gitignore`.
