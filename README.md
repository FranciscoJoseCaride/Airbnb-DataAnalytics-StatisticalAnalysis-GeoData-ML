# Airbnb_data
## Analisis de datos integral (descritivo, estadísitco, geo data, time series y machine learning)

En este repo se presenta un proyecto más integral. Con datos públicos de Airbnb (http://insideairbnb.com/get-the-data/ dataset de Madrid) hago un análisis descriptivo y estadístico, geoespacial, de series de tiempo, algo de nlp y ML para predecir el precio de una publicación. De manera adicional, dejo el código para correr parte del proyecto con Streamlit, una herramienta de python para hacer dashboards con pros y contras. El principal beneficio de Streamlit es su simpleza, con poco codigo python podes levantar un tablero en el browser, el paquete se hace cargo de todo (renderizar, webserver, etc). Una de las desventajas más grandes que le veo es, cada vez que lo actualizas o tocas corre todo desde 0 de nuevo y no es particularmente rápido, de hecho el dashboard que te dejo aca tarda un buen rato en levantarse por completo.

Archivos:
- descriptivo.ipynb es un análisis profundo del dataset, con datos estadísticos, series temporales, geodata, nlp, etc.
- dashboard.py es la misma info que descriptivo.ipynb pero en un tablero de Streamlit. Para levantarlo, después de instalarte el paquete, tenes que correr en consola streamlit run dashboard.py (ojo que tarda en levantar todo!)
- modelo.ipynb hago un poquito de ML para predecir la variable precio.
- carpeta dataset_nuevo son los datos provistos por Airbnb: http://insideairbnb.com/get-the-data/ dataset de Madrid.
- carpetas imagenes_prov y tablas_prov son archivos provisorios que se generan del dashboard.py para que corra. Acá las subo vacías.
- requirements.txt los paquetes necesarios para que corra todo. Desde la consola, en un ambiente virtual en blanco corre pip install -r requirements.txt
