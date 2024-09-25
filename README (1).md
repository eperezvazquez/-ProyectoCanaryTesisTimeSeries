# Explicacion del proyecto

En este proyecto los datos se obtienen de dos base de datos(BD), una BD una relacional sql SIGES y otra BD no relacional Mongo DB. Ambas fueron transformadas a SQL mediante pentaho de PDI. SIGES, sql es una base de datos que tiene más de 5 años de datos, y se busca poder predecir si un proyecto terminará en, fuera o por debajo de presupuesto en base a ciertas variables. Sobre este conjunto de datos son sobre portafolio de proyectos de AGESIC  datos desde el 2011 a la fecha. Estos datos surgen de una base de datos SQL que surgen de un sistema llamado SIGES.

¿Por qué elegimos realizar este proyecto?

Porque entendemos que con la aplicación de IA podremos contribuir en la mejora de los proyectos obteniendo una mejora en el proceso de gestión de los mismos y se traduzcan en una mejora que impacte a la ciudadanía.

Supuestos de Análisis
1) El análisis se realiza indistinto de las personas siguiendo los lineamientos Organizacionales.
2) Los proyectos posee un desvio que pasa a rojo mas del 20% y un desvio pasa amarillo mas del 10%, es un lineamiento con el que se trabaje.
3) La configuración de desvíos de tiempos si bien es un lineamiento se puede ajustar por proyecto es por ello que no lo tomamos en cuenta.
4) Un proyecto se entiende por desvío siguiendo los lineamientos organizacionales de desvío de presupuesto.

Asimismo se realizó el EDA completo donde se analizaron 362 columnas con 15778 registros, en ese proceso de análisis en base a datos estadísticos se subdividió el mismo en subdataset lo que permitió tener un análisis más exhaustivo del mismo. Lo que nos llevó a tener 

1)EDA Riesgos
2)EDA Lecciones Aprendidas
3)EDA Programa
4)EDA otros datos del proyecto
5)EDA Entregables
6)EDA Cronograma
7)EDA Presupuesto del proyecto
8)EDA Especifico modelo regresión
9)EDA Pagos Time Series

1)En el EDA de Riesgos encontramos:
Se analizaron 15 variables de las cuales representan 12952 riesgos. Lo motivos de los riesgos en base a una nube de palabra se puede detectar 13 riesgos que resaltan, a priori parecen pocos en base a la cantidad de registros asimismo la distribución de los riesgos parece con una distribución a conciencia donde el 37% son riesgos bajos, el 57% son riesgos medios y finalmente el 6% son riesgos altos. Asimismo cuando analizamos los riesgos altos el 14% de los proyectos poseen riesgos altos, Con respecto a los entregables son el 20.6%, posee una distribución pareto. Con respecto a la estrategia el 36.67% posee respuesta al riesgo el 23.67% contingencia, lo que se traduce que un 40% de los riesgos no posee respuesta o contingencia.

Recomendaciones
1. Mejorar el registro de riesgos más exhaustivo y con mayor detalle, dado que solo 13 representan los más aplicados.
2. El promedio de entre 7 y 8 riesgos por proyecto se visualiza pocos riesgos por proyectos.
3. Se recomienda tener individualizadas las oportunidades por alguna forma
4. Sería bueno tener una categorización de los riesgos estandarizada.
5. Se recomienda generar respuesta a los riesgos dado que el 40% no se tiene.
6. Mantener los criterios de ponderación de los riesgos dado que su distribución es adecuada.

2)EDA Lecciones Aprendidas

Resumen:

Se analizaron 15 variables con 12925 registros donde, el 61% de las lecciones aprendidas son de repetir lo que tienen una tendencia positiva y 39% evitar. De la nube de palabras se observa 5 que son las que se replican con 10 categorías de agrupación.

Recomendación: 
1. Se recomienda revisar la documentación de las lecciones aprendidas así como el proceso de generación de las mismas.

3) EDA de Programa

Se analizan 35 columnas de 12952 registros. Se obtiene que 87% de los programas estan habilitados y los programas con mayor cantidad de proyectos son:

1.Trámites
2.Implantación piloto 
3.Gobierno abierto
4.Expediente electrónico
5 E.Fondos
6.E gobierno
7.Administración Central

4)EDA otros datos del proyecto
Se  analizan 108 variables con 12098 registros. Del análisis de la matriz y de los números podemos decir que proyectos tiene una alta correlación con:
Entregables 0.97
Programa 0.99
Cronograma 0.99
Presupuesto 0.99
De análisis anteriores tenemos que
Riesgos 0.97
Lecciones Aprendidas 0.9

5)EDA Entregables
Se analizan 19 variables con 12098 registros. Se encuentra que el 60% de los proyectos poseen entregables mientras que el restante 40% no los posee. El 62.3% de los entregables son de tareas padres.
Recomienda:
1) Que todos los proyectos poseen entregables y/o productos definidos.

6) EDA Cronograma de proyecto
Se analizan 4 variables con 12098 registros. 
Para ver si un proyecto terminó en fecha o no, se concluyó que no se usa el semáforo sino la fecha inicio y fin con sus tolerancias pautadas para definir si termina en fecha.

7) EDA Presupuesto de proyecto
Se analizan 53 columnas de 12098 registros.  El 96.3 % registros son de presupuestos de áreas habilitadas mientras que el 3.7% es de áreas no habilitadas esto es porque en un momento estuvieron habilitadas y se dieron de baja por alguna razón.
La moneda que se toma en este caso no es la moneda del pago del proyecto sino que es la moneda de la adquisición lo que se recomienda revisar este tema.

8) EDA Específico modelo
Se analizaron 22 variables de 38488 registros. Los proyectos que terminaron con dificultad, entendiendo por ellos atraso en tiempo mayor al 20% y en costos sub o sobre ejecución y problemas de alcance que se ven en los riesgos, son 59.3% si tienen dificultades y el 40.7% no poseen dificultad.Si bien duracion posee oultliers no se ajusta dado que es una variable válida por la naturaleza del proyecto. 

9)EDA Pagos Time Series
Se analizaron 8 variables de 20238 cantidad de registros. 
Es importante conocer cómo es el comportamiento del sistema en cuanto al pago y sus adquisiciones, donde el monto se registra con la moneda original, la adquisición está definida en pesos, el pago sería en pesos, pero si la adquisición es en dólares, entonces ese monto del pago representa dólares.
El 67% de los pagos son en pesos. El 32% es en dolares, el 1% Euros. Se observa que incide años electorales, y la estacionalidad donde se concentra la mayor ejecucion en diciembre fin de año.Se puede observar el valor planificado en contraste con el valor real donde se observa que en el 2019 es año electoral, donde no hay presupuesto asignado lo cual afecta a la hora de desarrollar proyectos.


MODELOS APLICADOS

Para realizar estos modelos IA nos basamos en dos tipos de modelos un de clasificación, específicamente de regresion logistica y otro modelo de prophet.

En estadística, la regresión logística es un tipo de análisis de regresión utilizado para predecir el resultado de una variable categórica (una variable que puede adoptar un número limitado de categorías) en función de las variables independientes o predictoras. Es útil para modelar la probabilidad de un evento ocurriendo en función de otros factores. El análisis de regresión logística se enmarca en el conjunto de Modelos Lineales Generalizados (GLM por sus siglas en inglés) que usa como función de enlace la función logit. Las probabilidades que describen el posible resultado de un único ensayo se modelan como una función de variables explicativas, utilizando una función logística.

La regresión logística es un algoritmo de clasificación lineal. La clasificación es un problema en el que la tarea es asignar una categoría/clase a una nueva instancia que aprende las propiedades de cada clase a partir de los datos etiquetados existentes, llamado conjunto de entrenamiento. Ejemplos de problemas de clasificación pueden ser la clasificación de correos electrónicos como spam y no spam, observar la altura, el peso y otros atributos para clasificar a una persona como apta o no apta, etc.

En estadística, el modelo logístico (o modelo logit) se utiliza para modelar la probabilidad de que exista una determinada clase o evento, como pasa/falla, gana/pierde, vivo/muerto o sano/enfermo.

Modelo de Regresión Logística proyecto con dificultad

En este sentido en base al EDA que hicimos, las variables que inciden en proyectos con cronograma (tiempos), riesgos (alcance), costos (presupuesto), programa, entregables, lecciones aprendidas, fecha por el año en la cual se crea y duración. En ese sentido, es por ello que realizaremos un modelo predictivo que nos permita identificar si un proyecto terminará con dificultades o no, entendiendo por dificultades es: atraso en tiempo, sobre o sub ejecución de presupuesto, problemas con el alcance del mismo.  Para ello se realiza una análisis de las variables en base a la matriz de correlación.


Donde el valor de precisión dio 0.93, cuanto un proyecto si terminara con dificultad o no, en 1938 registros.
K-fold cross-validation results:
LogisticRegression average accuracy is 0.936
LogisticRegression average log_loss is 0.227
LogisticRegression average auc is 0.945


Modelo si un proyecto terminara fuera de presupuesto sub o sobre presupuesto.
En este caso tomando el EDA específico realizado, se procedió aplicar el modelo de regresión para saber si un proyecto termina con presupuesto correcto o no.

En este caso el valor de precisión es de 0.77 de un proyecto termina con presupuesto o no, pero en 264 registros. Es por ello que entre los dos modelos seleccionamos el que un proyecto termina con dificultad o no, dado que tiene una probabilidad mayor de acierto de 0.93 y en un universo mayor 1938 registros.

MODELO DE TIME SERIES
Ahora aplicaremos el modelo de time series para poder predecir cómo se comportará el presupuesto en pesos y dólares, esto permite ayudar a planificar mejor el mismo.

Aplicamos el modelo de Prophet que es una herramienta de código abierto de Facebook que se utiliza para pronosticar datos de series temporales que ayudan a las empresas a comprender y posiblemente predecir el mercado. Se basa en un modelo aditivo descomponible donde las tendencias no lineales se ajustan por estacionalidad, también tiene en cuenta los efectos de las vacaciones. Es por ello que aplicamos el mismo dado cuando analizamos los datos, tiene un comportamiento afectado por vacaciones y por estacionalidad. Asimismo realizamos dos modelos en pesos y en dolares, donde el escenario de dólares nos da mayor nivel de precisión nos quedamos con ambos modelos donde su nivel de ajuste es: interval_width=0.95, weekly_seasonality=False,seasonality_prior_scale=0.001,changepoint_prior_scale=0.9.

Por último hemos listado una serie de recomendaciones en cuanto al uso de SIGES y Wekan que se vuelcan del análisis:

Resumen de las recomendaciones asociadas al EDA 
1. Mejorar el registro de riesgos más exhaustivo y con mayor detalle, dado que solo 13 representan los más aplicados.
2. El promedio de entre 7 y 8 riesgos por proyecto se visualiza pocos riesgos por proyectos.
3. Se recomienda tener individualizadas las oportunidades por alguna forma
4. Sería bueno tener una categorización de los riesgos estandarizada.
5. Se recomienda generar respuesta a los riesgos dado que el 40% no se tiene.
6. Se recomienda revisar la documentación de las lecciones aprendidas asi como el proceso de generación de las mismas.
7. Que todos los proyectos poseen entregables y/o productos definidos.
8. La moneda que se toma en este caso no es la moneda del pago del proyecto sino que es la moneda de la adquisición lo que se recomienda revisar este tema.
9. Mantener los criterios de ponderación de los riesgos dado que su distribución es adecuada.
10. Todos los proyectos deberían tener descripciones, objetivos y beneficios
11. Que los tableros de wekan en Title board se asocien por el número de ID  con siges para mantener la traza.
12. Todos los proyectos tienen que tener una situación actual.
13. Uniformizar los criterios de los semaforos, amarillos y rojos en base a ciertas variables con el fin de que los datos se analizen sin sesgos.
14. Se recomienda cargar la estimación de presupuesto con el fin de tener una línea base del mismo dado que solo el 25% posee el mismo.
15. Se recomienda revisar la estructura de la tablas, por ejemplo pagos no tiene id moneda hay que ir por adquisición para obtener la misma.
16. Mejorar la calidad de los datos, por ejemplo lecciones aprendidas.
17. En el caso de los Riesgos mejorar la calidad muchas veces se incorporan de forma obligatoria y no terminan de reflejar los riesgos reales que sirven para una análisis de datos.
18. Mejoras le escala de presupuesto por ejemplo: Tipo Presupuesto	Subejecución	$ - Posible Subejecución Más de 30% de desviación
	Correcto	"$ - Ejecución dentro de lo Planificado Menos de 15 de desviación Sobreejecución	"$ - Posible Sobre-Ejecución
 Más de 20% de desviación"

Trabajos Futuros 
En resumen, tanto en la aplicación de Series Temporales a nivel de Índice General encontramos que la serie temporal estacional logra predecir mejor el modelo con un intervalo de confianza del 95% yweekly_seasonality=False y changepoint_prior_scale=0.9.Queda para analizar en trabajo futuros la aplicacion de un modelo, modelos MLP Multivariante para ver si se puede mejorar la prediccion.(https://www.youtube.com/watch?v=87c9D_41GWg)
En el caso del modelo de regresión queda para trabajos futuros aplicar random forest y ver su resultado.

Canary: https://canaryltisigeswekan.herokuapp.com/