🔧  "Análisis Estadístico y Predictivo de Respuesta al Tratamiento en Pacientes Oncológicos"

📊 Descripción del Proyecto

Introducción:
La respuesta al tratamiento en pacientes oncológicos es un factor crítico para evaluar la efectividad de las intervenciones terapéuticas. Este estudio combina análisis estadísticos descriptivos y métodos predictivos para identificar los factores clínicos y terapéuticos más relevantes asociados al éxito del tratamiento.

🎯 Objetivo
Analizar y predecir la respuesta al tratamiento en pacientes oncológicos mediante técnicas estadísticas avanzadas y modelos de aprendizaje automático, utilizando datos clínicos y terapéuticos.

Metodología:
Se analizó un conjunto de datos clínicos que incluye variables demográficas, características del tumor, tipo de tratamiento, y respuesta al mismo, clasificada en una variable binaria (éxito o fracaso). 
El análisis se dividió en tres fases principales:
1. Exploración y limpieza de datos, incluyendo la normalización de variables categóricas y el tratamiento de valores atípicos.
2. Análisis estadístico para evaluar asociaciones significativas entre las variables clínicas y la respuesta al tratamiento.Empleando testing AB basados en pruebas como chi-cuadrado dada la naturaleza de los datos,con el objetivo de e evaluar la relación general entre las variables, lo que incluye cualquier patrón de asociación (no solo una diferencia de proporciones).
   a) Relación entre Tumor Primario y Localización.
   b) Análisis entre Tratamiento Sistemico y Respuesta al tratamiento.
   c) Análisis entre DOSIS (Gy) (Cantidad de energía absorbida por un tejido debido a la radiación ionizante) y Respuesta       al tratamiento.
   d) Análisis entre Perfil Molecular y Resṕuesta al tratamiento.
   e) Analisis entre Cirugia Previa y Resṕuesta al tratamiento.
   f) Analisis de la relación entre la variable SRS y la respuesta al tratamiento 
   
4. Análisis multivariable.
5. Desarrollo de un modelo predictivo (regresión logística y random forest) para identificar factores clave asociados al éxito terapéutico y predecir la respuesta con métricas como precisión, sensibilidad y AUC-ROC.

Resultados esperados:
Se espera identificar variables significativas que influyen en la respuesta al tratamiento, como características del tumor, tipo de cirugía previa, o técnica de radioterapia utilizada. Además, los modelos predictivos desarrollados proporcionarán una herramienta útil para predecir el éxito terapéutico, optimizando la toma de decisiones clínicas.

Conclusión:
Este análisis combina técnicas estadísticas y de aprendizaje automático para aportar conocimiento sobre los factores determinantes en la respuesta al tratamiento en pacientes oncológicos, contribuyendo a mejorar la personalización de las intervenciones terapéuticas.




## Colaboradores 💻👨‍💻👩‍💻

- **Katia Berrios:**  Data Analyst [![`Linkedin`](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/katia-berrios/) [![`Github`](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/KtiaBM)
- **Melisa Rossi:** Data Scientist [![`Linkedin`](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/melisa-rossi-lagger/) [![`Github`](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/MelRossi)
- **Rosa González:** Data Scientist [![`Linkedin`](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rosa-isela-gonz%C3%A1lez-d%C3%ADaz/)[![`Github`](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/Rox-0864)
- **Ángel Troncoso:** Data Analyst [![`Linkedin`](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/angeltroncoso) [![`Github`](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/AngelTroncoso)

---

## Tecnologías Usadas 🛠️

- **Trello:** Herramienta de Gestión de Proyectos [![Trello](https://img.shields.io/badge/Trello-0079BF?logo=trello&logoColor=white)](https://trello.com/invite/b/66cd3c02fac81073b6752532/ATTI1258aad3b3bb787408fc3314244223832BFE00CD/s17-18-m-data-bi)
- **GitHub y Colab:** Desarrollo Colaborativo y Control de Versiones. [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/)
- **Slack:** Comunicación diaria del equipo y colaboración en tiempo real.[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://slack.com/)
- **Google Meet:** Reuniones diarias, planificación de sprint y coordinación de trabajo.[![Google Meet](https://img.shields.io/badge/Google%20Meet-00897B?logo=google-meet&logoColor=white)](https://meet.google.com/)
- **WhatsApp:** Comunicación instantánea para cuestiones urgentes.[![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?logo=whatsapp&logoColor=white)](https://www.whatsapp.com/)
- **Google Drive:** Almacenamiento y sincronización de documentación.[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=google-drive&logoColor=white)](https://drive.google.com/)

---

🔄 Flujo de Trabajo: metodologias Agiles Scrum y Kanban  

Definición del Experimento:

🔎 Identificación de las variables a probar (e.g., cambio de diseño, textos en botones).

🔄 División de grupos: Control y Tratamiento.

Recopilación de Datos:

💡 Simulación de interacciones de usuario con las dos versiones del producto.

📂 Almacenamiento en bases de datos estructuradas.

Análisis Estadístico:

🎯 Pruebas de hipótesis (e.g., t-test, chi-cuadrado).

🔢 Cálculo de métricas clave: tasa de conversión, tiempo promedio en la página, retención.

Visualización y Reporte:

💡 Creación de dashboards interactivos.

🗃️ Informe final con recomendaciones.

🔍 Estructura del Proyecto

```Simulado-S20-10-Data/

├── data/              # 📊 Datos simulados para pruebas A/B
├── notebooks/         # 📓 Notebooks de análisis y limpieza de datos
├── reports/           # 📑 Reportes finales y presentaciones
├── scripts/           # 🔧 Scripts de automatización y visualización
├── tests/             # 🔍 Pruebas unitarias y verificación de datos
└── README.md          # 📝 Descripción del proyecto
```

---
🔧 Requisitos Previos

🎓 Conocimiento básico de estadística y pruebas A/B.

Entorno configurado con:

Python 3.8+

Google Colaboratory (Colab)

Google Drive

🔄 Configuración del Entorno

Clonar este repositorio:

git clone https://github.com/No-Country-simulation/s20-10-data.git

Instalar las dependencias:

pip install -r requirements.txt

Configurar la base de datos PostgreSQL:

Crear una base de datos: no_contry_ab_testing

Ejecutar los scripts de inicialización en scripts/db_init.sql.

---

🕹️ Uso

📚 Cargar los datos simulados en la base de datos.

📓 Ejecutar los notebooks de análisis para obtener resultados iniciales.

📊 Generar reportes con visualizaciones y conclusiones.

---

🔄 Contribución

Este proyecto está abierto para mejoras y colaboraciones. Si deseas contribuir:

🔄 Haz un fork del repositorio.

🔧 Crea una rama para tu funcionalidad: git checkout -b feature/nueva-funcion.

📢 Envía un pull request con una descripción clara de los cambios.

📚 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.
