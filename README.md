Proyecto Simulado S20-10-Data: Testing A/B

Descripción del Proyecto

El proyecto Simulado-S20-10-Data está diseñado como una simulación laboral para la empresa ficticia No Contry, enfocada en el análisis de datos y pruebas A/B. El objetivo principal es mejorar la toma de decisiones basada en datos, utilizando experimentos controlados que permitan validar hipótesis de negocio y optimizar estrategias.

Objetivo

Realizar un análisis exhaustivo de los resultados de pruebas A/B implementadas en un entorno simulado para:

Evaluar el impacto de distintas variables en el comportamiento del usuario.

Determinar la versión más efectiva de un producto, servicio o interfaz.

Generar reportes accionables que respalden decisiones de negocio fundamentadas.

Tecnologías Utilizadas

Lenguajes: Python, R, SQL

Librerías y Herramientas:

Python: Pandas, NumPy, Matplotlib, SciPy, Seaborn

R: dplyr, ggplot2, tidyr

Bases de datos: PostgreSQL

Plataformas de visualización: Power BI, Tableau

Flujo de Trabajo

Definición del Experimento:

Identificación de las variables a probar (e.g., cambio de diseño, textos en botones).

División de grupos: Control y Tratamiento.

Recopilación de Datos:

Simulación de interacciones de usuario con las dos versiones del producto.

Almacenamiento en bases de datos estructuradas.

Análisis Estadístico:

Pruebas de hipótesis (e.g., t-test, chi-cuadrado).

Cálculo de métricas clave: tasa de conversión, tiempo promedio en la página, retención.

Visualización y Reporte:

Creación de dashboards interactivos.

Informe final con recomendaciones.

Estructura del Proyecto

Simulado-S20-10-Data/
|│
├── data/              # Datos simulados para pruebas A/B
├── notebooks/         # Notebooks de análisis y limpieza de datos
├── reports/           # Reportes finales y presentaciones
├── scripts/           # Scripts de automatización y visualización
├── tests/             # Pruebas unitarias y verificación de datos
└── README.md         # Descripción del proyecto

Requisitos Previos

Conocimiento básico de estadística y pruebas A/B.

Entorno configurado con:

Python 3.8+

R 4.0+

PostgreSQL

Configuración del Entorno

Clonar este repositorio:

git clone https://github.com/no-contry/simulado-s20-10-data.git

Instalar las dependencias:

pip install -r requirements.txt

Configurar la base de datos PostgreSQL:

Crear una base de datos: no_contry_ab_testing

Ejecutar los scripts de inicialización en scripts/db_init.sql.

Uso

Cargar los datos simulados en la base de datos.

Ejecutar los notebooks de análisis para obtener resultados iniciales.

Generar reportes con visualizaciones y conclusiones.

Contribución

Este proyecto está abierto para mejoras y colaboraciones. Si deseas contribuir:

Haz un fork del repositorio.

Crea una rama para tu funcionalidad: git checkout -b feature/nueva-funcion.

Envía un pull request con una descripción clara de los cambios.

Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.
