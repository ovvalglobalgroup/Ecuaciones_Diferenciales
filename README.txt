RETO N.º 3 — SIMULACIÓN IDS/IPS CON ECUACIONES DIFERENCIALES
========================================================

Aplicación de escritorio en Python para simular la respuesta dinámica
idealizada de un IDS/IPS ante malware en una tesorería blockchain
multifirma 5/5 mediante una ecuación diferencial de segundo orden

1. REQUISITOS
-------------
- Python 3 para Windows
- NumPy, SciPy, Matplotlib y Pillow
- Tkinter

Para instalar las bibliotecas, abre CMD en Windows y ejecuta:

    py -m pip install numpy scipy matplotlib pillow

2. ESTRUCTURA DEL PROYECTO
--------------------------
Carpeta del proyecto/
    index.py
    LEEME.txt
    Images/
        logo.png          (logo situado junto a PUCE)
        logo-app.png      (icono de la ventana)
    Reto_3_Resultados/    (se crea al exportar)

Los dos logotipos son opcionales, pero deben estar en la carpeta
Images para mostrarse en la aplicación

3. EJECUCIÓN
------------
Abre CMD en la carpeta que contiene index.py y ejecuta:

    py index.py

La aplicación se abre maximizada y carga los datos para los gráficos
No genera archivos automáticamente al iniciar

4. NAVEGACIÓN Y FUNCIONES
-------------------------
El menú lateral permite acceder a tres secciones:

    Amortiguamiento  — compara ζ = 0,35; 1,00; 1,80
    Rapidez          — compara ωₙ = 0,4; 0,8; 1,2 min⁻¹
    Demanda          — compara A = 0,6 y A = 1,0

Cada sección presenta un gráfico, una tabla con resultados numéricos,
la resolución matemática desarrollada paso a paso y una interpretación

Las fórmulas muestran fracciones verticales, potencias, raíces y
subíndices mediante MathText de Matplotlib

Controles:
    F11        Activa o desactiva pantalla completa
    Esc        Sale de pantalla completa o cierra el visor del gráfico
    Ctrl + E   Exporta los resultados
    Botón ⛶    Amplía el gráfico a pantalla completa

5. EXPORTACIÓN DE RESULTADOS
----------------------------
Pulsa «Exportar recursos» desde el menú lateral para crear la carpeta
Reto_3_Resultados y guardar en ella:

    Reto_3_Resultados/amortiguamiento.png
    Reto_3_Resultados/rapidez.png
    Reto_3_Resultados/demanda.png
    Reto_3_Resultados/resultados.xlsx

El libro resultados.xlsx contiene cuatro hojas:
    - Resumen: parámetros y métricas de los ocho escenarios
    - Amortiguamiento: evolución de las tres curvas
    - Rapidez: evolución de las tres curvas
    - Demanda: evolución de las dos curvas

Las hojas de curvas registran 1.501 muestras entre 0 y 30 minutos
Al volver a exportar se actualizan los archivos con los mismos nombres

La exportación CSV está desactivada en el código actual, por lo que no
se genera un nuevo resultados.csv al pulsar el botón
Si existe uno de versiones anteriores, puede permanecer en la carpeta

El botón «Abrir carpeta de resultados» permite acceder a los archivos
tras realizar la primera exportación

6. MODELO Y ALCANCE
-------------------
El programa integra numéricamente con solve_ivp y RK45 el modelo:

    x''(t) + 2ζωₙx'(t) + ωₙ²x(t) = ωₙ²A
    x(0) = 0; x'(0) = 0

También compara las soluciones numéricas con expresiones analíticas
y calcula el primer alcance del 95 %, el asentamiento ±5 % y el
sobreimpulso de cada escenario
