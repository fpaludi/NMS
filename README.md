# Introduccion

Se dispone la implementación de un pipeline de computer de Computer Vision que cuenta con una red Fast R-CNN pre-entrenada sobre el conjunto imágenes COCO, seguida por un algoritmo Non Maximum supression (NMS) standar.

Este modelo va a consumir todas las imágenes de la carpeta **data/images/** que tengan extension **.jpg** y va a escribir los resultados en el directorio **results** con extension **.png** (esto se debe que matplotlib no soporta  guardar imágenes .jpg, es algo a solucionar en el futuro con otras librerías). Para lograr esto, las carpetas **data** y **results** se montan como volúmenes en el container de docker

Para correr el modelo se debe ejecutar el siguiente comando

**docker-compose up --build -d**

Esto puede demorar unos minutos ya que hace uso de librerías pesadas como TensorFlow, OpenCV y Numpy.
Luego de unos minutos, las imágenes resultantes estarán listos en la carpeta **results**

# Consideraciones

El algoritmo NMS standar puede presentar errores a la hora de detectar objetos de la misma clase que están muy cercanos entre sí. Esto se debe a que las cajas o boxes de los mismos van a estar muy superpuestas y probablemente el algoritmo termine eliminando alguna de ellas.

Una alternativa a este problema es utilizar una versión modificada del algoritmo conocida como Soft-NMS. Este algoritmo no elimina las cajas que están muy superpuestas, sino que le reduce el score de las cajas cajas que tienen score más chico. Esto produce que el algoritmo pueda detectar objetos de la misma clase que están muy cercanos entre si, indicando menor confianza en algunos de ellos.


