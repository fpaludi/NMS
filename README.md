# Introduccion

## Sistema de CV
Se dispone la implementación de un pipeline de computer de Computer Vision que cuenta con una red Fast R-CNN pre-entrenada sobre el conjunto imágenes COCO, seguida por un algoritmo Non Maximum supression (NMS) standard.

Este modelo va a consumir todas las imágenes de la carpeta **data/images/** que tengan extension **.jpg** y va a escribir los resultados en el directorio **results** con extension **.png** (esto se debe que matplotlib no soporta  guardar imágenes .jpg, es algo a solucionar en el futuro con otras librerías). Para lograr esto, las carpetas **data** y **results** se montan como volúmenes en el container de docker. En el repositorio se encuentran hay disponibles 100 imágenes de ejemplo

Para correr el modelo se debe ejecutar el siguiente comando

**docker-compose up --build**

Esto puede demorar unos minutos ya que hace uso de librerías pesadas como TensorFlow, OpenCV y Numpy.
Luego de unos minutos, las imágenes resultantes estarán listos en la carpeta **results**

## Testeando algoritmo NMS 

Para validar el algoritmo se hicieron 2 tests básicos:
  * test_no_remove_box: Se crean cajas que no están superpuestas y se testea que todas las cajas "sobrevivan" al algoritmo NMS
  * test_remove_box: A un conjunto de cajas originales, se le agregan cajas con una superposición mayor a cierto valor (medida como IoU), se corre el algoritmo con un threshold menor al valor mencionado. Se verifica que solo "sobrevivan" las cajas originales

Para correr estos test se deben seguir los siguientes pasos:
```
sudo apt install git
sudo apt install python3-pip
sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_dev.txt
pytest -v
```

# Consideraciones

El algoritmo NMS standard puede introducir errores cunado procesa objetos de la misma clase que están muy cercanos entre sí. Esto se debe a que las cajas o boxes de los mismos van a estar muy superpuestas y el algoritmo puede terminar eliminando alguna de las cajas que deben permanecer en el resultado final

Una alternativa a este problema es utilizar una versión modificada del algoritmo conocida como Soft-NMS. Este algoritmo no elimina las cajas que están muy superpuestas, sino que le reduce el score de las cajas que tienen menor confianza. Esto produce que el algoritmo pueda detectar objetos de la misma clase que están muy cercanos entre si, pero indicando menor confianza en algunos de ellos.

# Detalles de implementación

Todo el pipeline de computer vision esta el archivo **model.py**. Mientras que las funciones que componen al algoritmo NMS estan dentro del archivo *nms.py** y son:
  * non_maximum_suppression
  * intersection_over_union

Donde **non_maximum_suppression** hace uso de **intersection_over_union**

# Pendientes

Quedo pendiente implementar el algoritmo Soft-NMS y ver el impacto de los threshold de ambos sobre algunas métricas de performance.

