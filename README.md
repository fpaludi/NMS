# Introduccion

Se dispone la implementación de un pipeline de computer de Computer Vision que cuenta con una red Fast R-CNN pre-entrenada sobre el conjunto imágenes COCO, seguida por un algoritmo Non Maximum supression (NMS) standard.

Este modelo va a consumir todas las imágenes de la carpeta **data/images/** que tengan extension **.jpg** y va a escribir los resultados en el directorio **results** con extension **.png** (esto se debe que matplotlib no soporta  guardar imágenes .jpg, es algo a solucionar en el futuro con otras librerías). Para lograr esto, las carpetas **data** y **results** se montan como volúmenes en el container de docker. En el repositorio se encuentran hay disponibles 100 imágenes de ejemplo

Para correr el modelo se debe ejecutar el siguiente comando

**docker-compose up --build**

Esto puede demorar unos minutos ya que hace uso de librerías pesadas como TensorFlow, OpenCV y Numpy.
Luego de unos minutos, las imágenes resultantes estarán listos en la carpeta **results**

# Consideraciones

El algoritmo NMS standard puede introducir errores cunado procesa objetos de la misma clase que están muy cercanos entre sí. Esto se debe a que las cajas o boxes de los mismos van a estar muy superpuestas y el algoritmo puede terminar eliminando alguna de las cajas que deben permanecer en el resultado final

Una alternativa a este problema es utilizar una versión modificada del algoritmo conocida como Soft-NMS. Este algoritmo no elimina las cajas que están muy superpuestas, sino que le reduce el score de las cajas que tienen menor confianza. Esto produce que el algoritmo pueda detectar objetos de la misma clase que están muy cercanos entre si, pero indicando menor confianza en algunos de ellos.


# Pendientes

Quedo pendiente implementar el algoritmo Soft-NMS y ver el impacto de los threshold de ambos sobre algunas métricas de performance.

