#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulación de la respuesta de un IDS/IPS ante malware en una tesorería blockchain multifirma 5/5


from __future__ import annotations

import csv
import math
import os
import sys
import matplotlib
import subprocess
import numpy as np

from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

matplotlib.use("Agg")  # Exportación de gráficos sin ventanas adicionales


# Rutas y tiempo de simulación
BASE = Path(__file__).resolve().parent
SALIDA = BASE / "Reto_3_Resultados"
TIEMPO_FINAL = 30.0
T = np.linspace(0.0, TIEMPO_FINAL, 1501)   # Muestras de tiempo cada 0,02 min

# Colores de la interfaz
# Paleta visual de la aplicación
INK = "#15283A"
TEAL = "#08778A"
BG = "#F3F6F8"
WHITE = "#FFFFFF"
MUTED = "#667587"
LINE = "#DDE6EC"
PINK = "#A43763"
BLUE = "#146C80"
GOLD = "#A87719"
# Parámetros de los tres escenarios
SERIES = {
    "amortiguamiento": [
        ("Bajo · ζ = 0,35", {"zeta": 0.35, "omega": 0.8, "demanda": 1.0}, PINK),
        ("Crítico · ζ = 1,00", {"zeta": 1.0, "omega": 0.8, "demanda": 1.0}, BLUE),
        ("Alto · ζ = 1,80", {"zeta": 1.8, "omega": 0.8, "demanda": 1.0}, GOLD),
    ],
    "rapidez": [
        ("ωₙ = 0,4 min⁻¹", {"zeta": 1.0, "omega": 0.4, "demanda": 1.0}, GOLD),
        ("ωₙ = 0,8 min⁻¹", {"zeta": 1.0, "omega": 0.8, "demanda": 1.0}, BLUE),
        ("ωₙ = 1,2 min⁻¹", {"zeta": 1.0, "omega": 1.2, "demanda": 1.0}, PINK),
    ],
    "demanda": [
        ("Demanda A = 0,6", {"zeta": 1.0, "omega": 0.8, "demanda": 0.6}, GOLD),
        ("Demanda A = 1,0", {"zeta": 1.0, "omega": 0.8, "demanda": 1.0}, BLUE),
    ],
}
# Títulos de los gráficos
TITULOS = {
    "amortiguamiento": "Efecto del amortiguamiento sobre la respuesta defensiva",
    "rapidez": "Efecto de la rapidez natural de respuesta (ζ = 1)",
    "demanda": "Efecto del nivel de demanda del ataque (ζ = 1; ωₙ = 0,8)",
}


# Método para resolver la ecuación diferencial con RK45
def simular(zeta: float, omega: float = 0.8, demanda: float = 1.0):
    """Resuelve x''+2ζωx'+ω²x=ω²A transformando y₁=x, y₂=x'."""
    if zeta <= 0 or omega <= 0 or demanda <= 0:
        raise ValueError("Los parámetros de estos escenarios deben ser positivos")

    # Método para calcular las derivadas del modelo
    def derivadas(_t, y):
        x, v = y
        return [v, omega**2 * (demanda - x) - 2.0 * zeta * omega * v]

    # Integración numérica con condiciones iniciales nulas
    sol = solve_ivp(
        derivadas, (float(T[0]), float(T[-1])), [0.0, 0.0], t_eval=T,
        method="RK45", rtol=1e-9, atol=1e-11
    )
    if not sol.success:
        raise RuntimeError("No fue posible resolver el modelo: " + sol.message)
    return sol.t, sol.y[0], sol.y[1]


# Método para validar la solución numérica
def solucion_analitica(t, zeta, omega, demanda):
    """Solución escalón exacta para contrastar la integración numérica."""
    t = np.asarray(t, dtype=float)
    # Caso subamortiguado
    if zeta < 1.0:
        s = math.sqrt(1 - zeta*zeta)
        return demanda * (1 - np.exp(-zeta*omega*t) * (
            np.cos(omega*s*t) + (zeta/s)*np.sin(omega*s*t)))
    # Caso de amortiguamiento crítico
    if math.isclose(zeta, 1.0):
        return demanda * (1 - (1 + omega*t)*np.exp(-omega*t))
    # Caso sobreamortiguado
    s = math.sqrt(zeta*zeta - 1)
    r1 = -omega*(zeta - s)
    r2 = -omega*(zeta + s)
    return demanda * (1 - (r2*np.exp(r1*t) - r1*np.exp(r2*t))/(r2-r1))


# Método para calcular tiempos y sobreimpulso
def metricas(t, x, demanda):
    """Mismas definiciones que el archivo original, sin interpolación subpaso."""
    indice_pico = int(np.argmax(x))
    pico = float(x[indice_pico])
    sobreimpulso = max(0.0, (pico/demanda - 1)*100)
    alcanza = np.where(x >= 0.95*demanda)[0]
    t95 = float(t[alcanza[0]]) if len(alcanza) else float("nan")
    fuera = np.where(np.abs(x-demanda) > 0.05*demanda)[0]
    asentamiento = (float(t[fuera[-1]+1])
                   if len(fuera) and fuera[-1] < len(t)-1 else float("nan"))
    return {
        "pico": pico, "t_pico": float(t[indice_pico]),
        "sobreimpulso_pct": sobreimpulso, "t95_min": t95,
        "asentamiento_5pct_min": asentamiento,
    }


# Método para preparar los escenarios de simulación
def calcular_datos():
    """Calcula una sola vez todas las curvas para la GUI y las exportaciones."""
    datos = {}
    # Recorrer los escenarios y validar las soluciones
    for clave, configuraciones in SERIES.items():
        filas = []
        for etiqueta, parametros, color in configuraciones:
            t, x, velocidad = simular(**parametros)
            exacta = solucion_analitica(t, **parametros)
            error = float(np.max(np.abs(x - exacta)))
            if error > 2e-6:
                raise RuntimeError(f"La validación analítica falló: {etiqueta}, error={error}")
            filas.append({
                "etiqueta": etiqueta, "parametros": parametros, "color": color,
                "t": t, "x": x, "v": velocidad, "metricas": metricas(t, x, parametros["demanda"]),
                "error_validacion": error,
            })
        datos[clave] = filas
    return datos


# Método para construir los gráficos
def construir_figura(clave, filas):
    tam = (8.6, 3.65) if clave == "amortiguamiento" else (8.6, 3.5)
    fig = Figure(figsize=tam, dpi=115, facecolor="white")
    ax = fig.add_subplot(111)
    # Dibujar cada curva según sus parámetros
    for fila in filas:
        ax.plot(fila["t"], fila["x"], label=fila["etiqueta"],
                color=fila["color"], linewidth=2.1)
        if clave == "demanda":
            ax.axhline(fila["parametros"]["demanda"], color=fila["color"],
                       linestyle=":", linewidth=1.0, alpha=0.8)
    if clave != "demanda":
        ax.axhline(1.0, linestyle="--", color="#62676C", linewidth=1.1,
                   label="Demanda de referencia (A = 1)" if clave == "amortiguamiento" else None)
    ax.set_xlim(0, 22 if clave == "amortiguamiento" else 19 if clave == "rapidez" else 18)
    ax.set_ylim(0, 1.45 if clave == "amortiguamiento" else 1.10 if clave == "rapidez" else 1.12)
    ax.set_title(TITULOS[clave], fontsize=12, weight="bold", pad=13, color=INK)
    ax.grid(True, color="#E7ECF0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Tiempo desde el inicio del ataque (min)")
    ax.set_ylabel("Índice de respuesta x(t)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout(pad=1.5)
    return fig


# Método para crear el archivo CSV
def escribir_csv(datos):
    campos = ["escenario", "zeta", "omega_min-1", "demanda", "pico", "t_pico",
              "sobreimpulso_pct", "t95_min", "asentamiento_5pct_min"]
    destino = SALIDA / "resultados.csv"
    # Escritura del resumen en CSV
    with destino.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for clave, filas in datos.items():
            for fila in filas:
                p = fila["parametros"]
                writer.writerow({
                    "escenario": f"{clave}: {fila['etiqueta']}",
                    "zeta": p["zeta"], "omega_min-1": p["omega"], "demanda": p["demanda"],
                    **fila["metricas"]
                })
    return destino


# Método para generar celdas en Excel
def celda_excel(valor, indice_col, indice_fila, encabezado=False):
    """Celda OOXML de Excel sin bibliotecas adicionales."""
    col = ""
    n = indice_col
    while n:
        n, resto = divmod(n-1, 26)
        col = chr(65 + resto) + col
    ref = f"{col}{indice_fila}"
    sty = ' s="1"' if encabezado else ' s="2"' if isinstance(valor, (int, float)) else ''
    if valor is None or (isinstance(valor, (int, float)) and not math.isfinite(float(valor))):
        return f'<c r="{ref}"{sty}/>'
    if isinstance(valor, (int, float)):
        return f'<c r="{ref}"{sty}><v>{float(valor):.12g}</v></c>'
    return f'<c r="{ref}" s="{1 if encabezado else 0}" t="inlineStr"><is><t>{escape(str(valor))}</t></is></c>'


# Método para crear el libro Excel
def escribir_xlsx(datos):
    """Libro XLSX legible por Excel, creado con biblioteca estándar zipfile/XML.

    Incluye Resumen y las tres series temporales completas (paso de 0,02 min).
    No se requiere openpyxl ni otro paquete adicional en la PC del estudiante.
    """
    destino = SALIDA / "resultados.xlsx"
    columnas = ["Escenario", "ζ", "ωₙ (min⁻¹)", "A", "Pico", "Tiempo pico (min)",
                "Sobreimpulso (%)", "Primer 95 % (min)", "Estabilización ±5 % (min)"]
    resumen = [columnas]
    for clave, filas in datos.items():
        for fila in filas:
            p, m = fila["parametros"], fila["metricas"]
            resumen.append([f"{clave} · {fila['etiqueta']}", p["zeta"], p["omega"],
                            p["demanda"], m["pico"], m["t_pico"],
                            m["sobreimpulso_pct"], m["t95_min"], m["asentamiento_5pct_min"]])

    # Crear el resumen y las hojas de las curvas
    hojas = [("Resumen", resumen)]
    for clave, filas in datos.items():
        encabezado = ["Tiempo (min)"] + [f["etiqueta"] for f in filas]
        registros = [encabezado]
        for k in range(len(T)):
            registros.append([float(T[k])] + [float(f["x"][k]) for f in filas])
        hojas.append((clave.capitalize(), registros))

    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                  for i in range(1, len(hojas)+1)) + '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>')
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                  for i in range(1, len(hojas)+1))
        + f'<Relationship Id="rId{len(hojas)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>')
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>' + ''.join(f'<sheet name="{escape(nombre)}" sheetId="{i}" r:id="rId{i}"/>'
                             for i, (nombre, _) in enumerate(hojas, 1))
        + '</sheets></workbook>')
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>
  <fills count="3"><fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF08778A"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/>
    <xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    # Empaquetar el contenido del archivo XLSX
    with ZipFile(destino, "w", ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/styles.xml", styles)
        for i, (nombre, filas) in enumerate(hojas, 1):
            # Anchos de columnas en Excel
            col_widths = ([43] + [20]*(len(filas[0])-1)) if i == 1 else [19] + [29]*(len(filas[0])-1)
            cols = ''.join(f'<col min="{j}" max="{j}" width="{w}" customWidth="1"/>'
                           for j, w in enumerate(col_widths, 1))
            rows = []
            for j, fila in enumerate(filas, 1):
                cells = ''.join(celda_excel(valor, c, j, encabezado=(j == 1))
                                for c, valor in enumerate(fila, 1))
                rows.append(f'<row r="{j}"{(" ht=\"26\" customHeight=\"1\"" if j == 1 else "")}>{cells}</row>')
            hoja = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
                'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
                f'<cols>{cols}</cols><sheetData>{"".join(rows)}</sheetData>'
                f'<autoFilter ref="A1:{chr(64+len(filas[0]))}{len(filas)}"/>'
                '</worksheet>')
            z.writestr(f"xl/worksheets/sheet{i}.xml", hoja)
    return destino


# Método para exportar imágenes y Excel
def exportar(datos):
    SALIDA.mkdir(parents=True, exist_ok=True)
    rutas = []
    for i, clave in enumerate(SERIES, 1):
        figura = construir_figura(clave, datos[clave])
        ruta = SALIDA / f"{clave}.png"
        figura.savefig(ruta, dpi=220, facecolor="white")
        rutas.append(ruta)
    # Exportación CSV desactivada
    # rutas.append(escribir_csv(datos))
    rutas.append(escribir_xlsx(datos))
    return rutas


# Método para formatear valores numéricos
def fnum(v, dec=2):
    return "—" if not math.isfinite(v) else f"{v:.{dec}f}".replace(".", ",")


# Método para desarrollar todas las soluciones matemáticas sin abreviar
# Las fórmulas se renderizan con MathText usando el prefijo de fórmula

def pasos_de_resolucion(clave, filas):
    """Explica el planteamiento, las raíces, las constantes y cada solución"""

    def M(formula):
        # Marcar las expresiones que se dibujarán como ecuaciones matemáticas
        return "@FORMULA:" + formula

    def B(*lineas):
        # Unir la explicación y las expresiones de cada tarjeta
        return "\n".join(lineas)

    intro = [
        ("1 · MODELO Y CONDICIONES INICIALES", B(
            "Se modela la respuesta idealizada x(t) de un IDS/IPS mediante una EDO lineal de segundo orden:",
            M(r"x''(t)+2\zeta\omega_n x'(t)+\omega_n^{2}x(t)=\omega_n^{2}A"),
            "El sistema parte sin respuesta acumulada ni velocidad inicial:",
            M(r"x(0)=0,\qquad x'(0)=0"),
            "ζ es el amortiguamiento, ωₙ determina la rapidez natural y A representa la demanda normalizada",
            "x(t) es un índice didáctico, no una medición real de seguridad"
        )),
        ("2 · CONVERSIÓN PARA LA SIMULACIÓN CON RK45", B(
            "Primero se despeja la derivada de segundo orden:",
            M(r"x''=\omega_n^{2}(A-x)-2\zeta\omega_n x'"),
            "Se introducen dos variables de estado:",
            M(r"y_1=x,\qquad y_2=x'"),
            "Por definición, la derivada de la primera variable es la segunda:",
            M(r"y'_1=y_2"),
            "La segunda derivada se sustituye por la ecuación despejada:",
            M(r"y'_2=\omega_n^{2}(A-y_1)-2\zeta\omega_n y_2"),
            "Las condiciones iniciales del sistema equivalente son:",
            M(r"y_1(0)=0,\qquad y_2(0)=0"),
            "Python integra las dos ecuaciones con solve_ivp y RK45 entre 0 y 30 minutos"
        )),
    ]

    if clave == "amortiguamiento":
        return intro + [
            ("3 · MÉTODO GENERAL PARA LA SOLUCIÓN ANALÍTICA", B(
                "La ecuación homogénea se obtiene al igualar a cero el término de entrada:",
                M(r"x''+2\zeta\omega_nx'+\omega_n^2x=0"),
                "Se propone una solución exponencial para la parte homogénea:",
                M(r"x_h(t)=e^{rt}"),
                "Al sustituirla se obtiene la ecuación característica:",
                M(r"r^2+2\zeta\omega_n r+\omega_n^2=0"),
                "La fórmula cuadrática permite identificar las raíces:",
                M(r"r=\frac{-2\zeta\omega_n\pm\sqrt{(2\zeta\omega_n)^2-4\omega_n^2}}{2}"),
                "Como la entrada es constante, se ensaya una solución particular constante K:",
                M(r"\omega_n^2K=\omega_n^2A\quad\Longrightarrow\quad K=A"),
                "La solución total es la suma de la solución particular y la homogénea",
                "En los tres casos se mantienen ωₙ = 0,8 min⁻¹ y A = 1"
            )),
            ("4 · AMORTIGUAMIENTO BAJO: ECUACIÓN Y RAÍCES", B(
                "Se sustituye ζ = 0,35, ωₙ = 0,8 y A = 1:",
                M(r"x''+2(0{,}35)(0{,}8)x'+(0{,}8)^2x=(0{,}8)^2"),
                "Después de multiplicar y elevar al cuadrado:",
                M(r"x''+0{,}56x'+0{,}64x=0{,}64"),
                "La ecuación característica de la parte homogénea es:",
                M(r"r^2+0{,}56r+0{,}64=0"),
                "Se calcula el discriminante:",
                M(r"\Delta=(0{,}56)^2-4(0{,}64)=-2{,}2464"),
                "Al ser negativo, aparecen dos raíces complejas conjugadas:",
                M(r"r=\frac{-0{,}56\pm\sqrt{-2{,}2464}}{2}"),
                M(r"r_{1,2}=-0{,}28\pm0{,}7494\mathrm{i}"),
                "La parte imaginaria coincide con la frecuencia amortiguada:",
                M(r"\omega_d=0{,}8\sqrt{1-(0{,}35)^2}\approx0{,}7494")
            )),
            ("5 · AMORTIGUAMIENTO BAJO: CONSTANTES Y SOLUCIÓN", B(
                "Las raíces complejas producen una combinación de seno y coseno:",
                M(r"x(t)=1+e^{-0{,}28t}[C_1\cos(\omega_dt)+C_2\sin(\omega_dt)]"),
                "Al imponer x(0) = 0 y usar cos(0) = 1, sen(0) = 0:",
                M(r"0=1+C_1\quad\Longrightarrow\quad C_1=-1"),
                "Se deriva la función mediante la regla del producto y la regla de la cadena:",
                M(r"x'(t)=e^{-0{,}28t}[-0{,}28(C_1\cos(\omega_dt)+C_2\sin(\omega_dt))]"),
                M(r"\qquad+e^{-0{,}28t}[-C_1\omega_d\sin(\omega_dt)+C_2\omega_d\cos(\omega_dt)]"),
                "La derivada en t = 0 proporciona la segunda condición:",
                M(r"x'(0)=-0{,}28C_1+\omega_dC_2=0"),
                "Se sustituye C₁ = −1 y se despeja C₂:",
                M(r"0=0{,}28+0{,}7494C_2"),
                M(r"C_2=-\frac{0{,}28}{0{,}7494}\approx-0{,}3736"),
                "Al reemplazar las constantes se obtiene la respuesta:",
                M(r"x(t)\approx1-e^{-0{,}28t}[\cos(\omega_dt)+0{,}3736\sin(\omega_dt)]"),
                "La solución oscila antes de acercarse al valor de referencia A = 1"
            )),
            ("6 · AMORTIGUAMIENTO CRÍTICO: RAÍZ DOBLE", B(
                "Se fija ζ = 1, ωₙ = 0,8 y A = 1:",
                M(r"x''+2(1)(0{,}8)x'+(0{,}8)^2x=(0{,}8)^2"),
                M(r"x''+1{,}6x'+0{,}64x=0{,}64"),
                "La ecuación característica se factoriza:",
                M(r"r^2+1{,}6r+0{,}64=(r+0{,}8)^2=0"),
                "Se obtiene una raíz doble:",
                M(r"r_1=r_2=-0{,}8"),
                "Para una raíz doble, la solución total toma esta forma:",
                M(r"x(t)=1+(C_1+C_2t)e^{-0{,}8t}"),
                "De x(0) = 0 se encuentra la primera constante:",
                M(r"0=1+C_1\quad\Longrightarrow\quad C_1=-1"),
                "Se deriva mediante la regla del producto:",
                M(r"x'(t)=[C_2-0{,}8(C_1+C_2t)]e^{-0{,}8t}"),
                "De x'(0) = 0 y C₁ = −1 se obtiene:",
                M(r"0=C_2-0{,}8C_1=C_2+0{,}8"),
                M(r"C_2=-0{,}8"),
                "Finalmente, la solución particular es:",
                M(r"x(t)=1-(1+0{,}8t)e^{-0{,}8t}")
            )),
            ("7 · AMORTIGUAMIENTO ALTO: DOS RAÍCES REALES", B(
                "Se fija ζ = 1,8, ωₙ = 0,8 y A = 1:",
                M(r"x''+2(1{,}8)(0{,}8)x'+(0{,}8)^2x=(0{,}8)^2"),
                M(r"x''+2{,}88x'+0{,}64x=0{,}64"),
                "Se plantea la ecuación característica:",
                M(r"r^2+2{,}88r+0{,}64=0"),
                "Se calcula el discriminante y se aplica la fórmula cuadrática:",
                M(r"\Delta=(2{,}88)^2-4(0{,}64)=5{,}7344"),
                M(r"r=\frac{-2{,}88\pm\sqrt{5{,}7344}}{2}"),
                M(r"r_1\approx-0{,}2427,\qquad r_2\approx-2{,}6373"),
                "La solución total combina dos exponentes reales distintos:",
                M(r"x(t)=1+C_1e^{r_1t}+C_2e^{r_2t}"),
                "La derivada de esta solución es:",
                M(r"x'(t)=r_1C_1e^{r_1t}+r_2C_2e^{r_2t}"),
                "Al aplicar las dos condiciones iniciales se obtiene el sistema:",
                M(r"C_1+C_2=-1"),
                M(r"r_1C_1+r_2C_2=0"),
                "De la primera ecuación se despeja C₂ = −1 − C₁ y se sustituye:",
                M(r"r_1C_1+r_2(-1-C_1)=0"),
                M(r"C_1=\frac{r_2}{r_1-r_2}\approx-1{,}1013"),
                M(r"C_2=-1-C_1\approx0{,}1013"),
                "Se obtiene la solución particular:",
                M(r"x(t)=1-\frac{r_2e^{r_1t}-r_1e^{r_2t}}{r_2-r_1}"),
                "Ambas raíces son negativas y la respuesta no presenta oscilaciones"
            )),
            ("8 · COMPROBACIÓN NUMÉRICA E INTERPRETACIÓN", B(
                "Las fórmulas anteriores se evalúan con las condiciones iniciales:",
                M(r"x(0)=0,\quad x'(0)=0,\quad\lim_{t\to\infty}x(t)=1"),
                "Para calcular el tiempo al 95 %, Python busca el primer instante con:",
                M(r"x(t)\geq0{,}95A"),
                "El asentamiento ±5 % exige permanecer en el intervalo:",
                M(r"0{,}95A\leq x(t)\leq1{,}05A"),
                "Bajo (ζ = 0,35): alcanza el 95 % en 2,46 min y se asienta en 9,86 min; sobreimpulso 30,9 %",
                "Crítico (ζ = 1): alcanza el 95 % y se asienta en 5,94 min; sin sobreimpulso",
                "Alto (ζ = 1,8): alcanza el 95 % y se asienta en 12,76 min; sin sobreimpulso",
                "Los valores temporales proceden de la integración RK45, no de redondear raíces"
            )),
        ]

    if clave == "rapidez":
        return intro + [
            ("3 · ECUACIÓN GENERAL CON AMORTIGUAMIENTO CRÍTICO", B(
                "Se mantiene ζ = 1 y A = 1 y se modifica la frecuencia natural ωₙ:",
                M(r"x''+2\omega_nx'+\omega_n^2x=\omega_n^2"),
                "La ecuación característica se factoriza:",
                M(r"r^2+2\omega_nr+\omega_n^2=(r+\omega_n)^2=0"),
                "Existe una raíz doble para cada valor de ωₙ:",
                M(r"r_1=r_2=-\omega_n"),
                "La entrada constante admite la solución particular xₚ = 1:",
                M(r"x(t)=1+(C_1+C_2t)e^{-\omega_nt}"),
                "Al imponer x(0) = 0 se obtiene C₁ = −1:",
                M(r"0=1+C_1\quad\Longrightarrow\quad C_1=-1"),
                "Se deriva la expresión general:",
                M(r"x'(t)=[C_2-\omega_n(C_1+C_2t)]e^{-\omega_nt}"),
                "Se impone x'(0) = 0 y se despeja la segunda constante:",
                M(r"0=C_2-\omega_nC_1\quad\Longrightarrow\quad C_2=-\omega_n"),
                "La solución general para cualquier ωₙ positivo queda:",
                M(r"x(t)=1-(1+\omega_nt)e^{-\omega_nt}")
            )),
            ("4 · RAPIDEZ BAJA: FRECUENCIA 0,4", B(
                "Se sustituye ωₙ = 0,4 min⁻¹ en la EDO:",
                M(r"x''+2(0{,}4)x'+(0{,}4)^2x=(0{,}4)^2"),
                M(r"x''+0{,}8x'+0{,}16x=0{,}16"),
                "Se factoriza la ecuación característica:",
                M(r"r^2+0{,}8r+0{,}16=(r+0{,}4)^2=0"),
                M(r"r_1=r_2=-0{,}4"),
                "Con las condiciones iniciales se calculan las constantes:",
                M(r"C_1=-1,\qquad C_2=-0{,}4"),
                "Solución particular:",
                M(r"x(t)=1-(1+0{,}4t)e^{-0{,}4t}"),
                "El modelo alcanza el 95 % en 11,86 min"
            )),
            ("5 · RAPIDEZ MEDIA: FRECUENCIA 0,8", B(
                "Se sustituye ωₙ = 0,8 min⁻¹ en la EDO:",
                M(r"x''+2(0{,}8)x'+(0{,}8)^2x=(0{,}8)^2"),
                M(r"x''+1{,}6x'+0{,}64x=0{,}64"),
                "Se factoriza la ecuación característica:",
                M(r"r^2+1{,}6r+0{,}64=(r+0{,}8)^2=0"),
                M(r"r_1=r_2=-0{,}8"),
                "Con las condiciones iniciales se calculan las constantes:",
                M(r"C_1=-1,\qquad C_2=-0{,}8"),
                "Solución particular:",
                M(r"x(t)=1-(1+0{,}8t)e^{-0{,}8t}"),
                "El modelo alcanza el 95 % en 5,94 min"
            )),
            ("6 · RAPIDEZ ALTA: FRECUENCIA 1,2", B(
                "Se sustituye ωₙ = 1,2 min⁻¹ en la EDO:",
                M(r"x''+2(1{,}2)x'+(1{,}2)^2x=(1{,}2)^2"),
                M(r"x''+2{,}4x'+1{,}44x=1{,}44"),
                "Se factoriza la ecuación característica:",
                M(r"r^2+2{,}4r+1{,}44=(r+1{,}2)^2=0"),
                M(r"r_1=r_2=-1{,}2"),
                "Con las condiciones iniciales se calculan las constantes:",
                M(r"C_1=-1,\qquad C_2=-1{,}2"),
                "Solución particular:",
                M(r"x(t)=1-(1+1{,}2t)e^{-1{,}2t}"),
                "El modelo alcanza el 95 % en 3,96 min"
            )),
            ("7 · VALIDACIÓN Y COMPARACIÓN DE RAPIDEZ", B(
                "Cada curva se compara con la solución numérica calculada por RK45",
                "El tiempo al 95 % se obtiene al buscar el primer instante que cumple:",
                M(r"1-(1+\omega_nt)e^{-\omega_nt}\geq0{,}95"),
                "Se usa el mismo intervalo de 0 a 30 minutos y las mismas condiciones iniciales",
                "ωₙ = 0,4: t₉₅ = 11,86 min; ωₙ = 0,8: t₉₅ = 5,94 min; ωₙ = 1,2: t₉₅ = 3,96 min",
                "Las tres respuestas son críticas, monótonas y no presentan sobreimpulso",
                "En este modelo, aumentar ωₙ acorta el tiempo de respuesta"
            )),
        ]

    return intro + [
        ("3 · MÉTODO GENERAL CON DEMANDA VARIABLE", B(
            "Se mantienen ζ = 1 y ωₙ = 0,8 min⁻¹ y se modifica la demanda A:",
            M(r"x''+1{,}6x'+0{,}64x=0{,}64A"),
            "La ecuación característica homogénea no depende de A:",
            M(r"r^2+1{,}6r+0{,}64=(r+0{,}8)^2=0"),
            M(r"r_1=r_2=-0{,}8"),
            "Para encontrar la solución particular se supone xₚ(t) = K:",
            M(r"0{,}64K=0{,}64A\quad\Longrightarrow\quad K=A"),
            "La solución general se forma con una raíz doble:",
            M(r"x(t)=A+(C_1+C_2t)e^{-0{,}8t}"),
            "Con x(0) = 0 se despeja la primera constante:",
            M(r"0=A+C_1\quad\Longrightarrow\quad C_1=-A"),
            "Se deriva la solución general:",
            M(r"x'(t)=[C_2-0{,}8(C_1+C_2t)]e^{-0{,}8t}"),
            "Al imponer x'(0) = 0 se calcula C₂:",
            M(r"0=C_2-0{,}8C_1\quad\Longrightarrow\quad C_2=-0{,}8A"),
            "La solución válida para cualquier A positivo es:",
            M(r"x(t)=A[1-(1+0{,}8t)e^{-0{,}8t}]")
        )),
        ("4 · DEMANDA BAJA: A = 0,6", B(
            "Se sustituye A = 0,6 en el lado derecho de la EDO:",
            M(r"x''+1{,}6x'+0{,}64x=0{,}64(0{,}6)"),
            M(r"x''+1{,}6x'+0{,}64x=0{,}384"),
            "La parte homogénea mantiene la raíz doble −0,8",
            "La solución particular constante es K = 0,6:",
            M(r"x(t)=0{,}6+(C_1+C_2t)e^{-0{,}8t}"),
            "Con la primera condición inicial se obtiene:",
            M(r"0=0{,}6+C_1\quad\Longrightarrow\quad C_1=-0{,}6"),
            "Con la segunda condición inicial:",
            M(r"0=C_2-0{,}8(-0{,}6)=C_2+0{,}48"),
            M(r"C_2=-0{,}48"),
            "La solución particular queda:",
            M(r"x(t)=0{,}6-(0{,}6+0{,}48t)e^{-0{,}8t}"),
            "Forma factorizada equivalente:",
            M(r"x(t)=0{,}6[1-(1+0{,}8t)e^{-0{,}8t}]"),
            "El 95 % de la demanda equivale al umbral 0,57:",
            M(r"0{,}95(0{,}6)=0{,}57")
        )),
        ("5 · DEMANDA DE REFERENCIA: A = 1", B(
            "Se sustituye A = 1 en la ecuación:",
            M(r"x''+1{,}6x'+0{,}64x=0{,}64(1)"),
            M(r"x''+1{,}6x'+0{,}64x=0{,}64"),
            "La solución particular constante es K = 1:",
            M(r"x(t)=1+(C_1+C_2t)e^{-0{,}8t}"),
            "Con x(0) = 0 se encuentra C₁:",
            M(r"0=1+C_1\quad\Longrightarrow\quad C_1=-1"),
            "Con x'(0) = 0 se encuentra C₂:",
            M(r"0=C_2-0{,}8(-1)=C_2+0{,}8"),
            M(r"C_2=-0{,}8"),
            "La solución particular resulta:",
            M(r"x(t)=1-(1+0{,}8t)e^{-0{,}8t}"),
            "En este caso el umbral del 95 % es 0,95:",
            M(r"0{,}95(1)=0{,}95")
        )),
        ("6 · COMPARACIÓN DE LAS DOS DEMANDAS", B(
            "Para comparar ambas curvas se divide la respuesta entre la demanda:",
            M(r"\frac{x(t)}{A}=1-(1+0{,}8t)e^{-0{,}8t}"),
            "La expresión normalizada no depende de A, por eso ambas curvas tienen los mismos tiempos relativos",
            "El tiempo al 95 % se determina resolviendo numéricamente:",
            M(r"1-(1+0{,}8t)e^{-0{,}8t}\geq0{,}95"),
            "Para A = 0,6 y A = 1, el primer 95 % y el asentamiento ±5 % ocurren en 5,94 min",
            "No existe sobreimpulso en ninguno de los dos escenarios críticos",
            "La amplitud cambia proporcionalmente con A, pero no la rapidez relativa",
            "Este modelo lineal no representa saturación ni sobrecarga real del IDS/IPS"
        )),
    ]


# Fórmulas matemáticas para los pasos de resolución
# Potencias, raíces, subíndices y fracciones
# Representación visual con MathText
# Texto explicativo antes y después de cada fórmula
# Correspondencia entre texto y ecuaciones formateadas
MATEMATICA_VISUAL = {
    "Modelo: x″(t) + 2ζωₙ x′(t) + ωₙ² x(t) = ωₙ² A.": (
        "Modelo:", r"x''(t)+2\zeta\omega_n x'(t)+\omega_n^{2}x(t)=\omega_n^{2}A", ""),
    "Condiciones iniciales: x(0) = 0 y x′(0) = 0.": (
        "Condiciones iniciales:", r"x(0)=0,\qquad x'(0)=0", ""),
    "x″(t) = ωₙ² [A − x(t)] − 2ζωₙ x′(t)": (
        "", r"x''(t)=\omega_n^{2}\left[A-x(t)\right]-2\zeta\omega_n x'(t)", ""),
    "Definiendo y₁ = x(t), y₂ = x′(t):": (
        "Definiendo:", r"y_1=x(t),\qquad y_2=x'(t)", ""),
    "y₁′ = y₂": ("", r"y'_1=y_2", ""),
    "y₂′ = ωₙ² (A − y₁) − 2ζωₙ y₂": (
        "", r"y'_2=\omega_n^{2}(A-y_1)-2\zeta\omega_n y_2", ""),
    "Con ωₙ = 0,8 y A = 1: x″ + (1,6ζ)x′ + 0,64x = 0,64.": (
        "Con ωₙ = 0,8 y A = 1:", r"x''+1{,}6\zeta x'+0{,}64x=0{,}64", ""),
    "• ζ = 0,35 → x″ + 0,56x′ + 0,64x = 0,64; raíces complejas: oscilación amortiguada.": (
        "• ζ = 0,35 →", r"x''+0{,}56x'+0{,}64x=0{,}64", "Raíces complejas: oscilación amortiguada."),
    "• ζ = 1,00 → x″ + 1,60x′ + 0,64x = 0,64; raíz doble −0,8: respuesta crítica.": (
        "• ζ = 1,00 →", r"x''+1{,}60x'+0{,}64x=0{,}64", "Raíz doble −0,8: respuesta crítica."),
    "• ζ = 1,80 → x″ + 2,88x′ + 0,64x = 0,64; dos raíces reales negativas.": (
        "• ζ = 1,80 →", r"x''+2{,}88x'+0{,}64x=0{,}64", "Dos raíces reales negativas."),
    "Para ζ < 1, sea ωd = ωₙ√(1−ζ²):": (
        "Para ζ < 1, sea:", r"\omega_d=\omega_n\sqrt{1-\zeta^2}", ""),
    "x(t) = A[1 − e^(−ζωₙt){cos(ωd t) + ζ sen(ωd t)/√(1−ζ²)}].": (
        "", r"x(t)=A\left[1-e^{-\zeta\omega_n t}\left(\cos(\omega_d t)+\frac{\zeta\sin(\omega_d t)}{\sqrt{1-\zeta^2}}\right)\right]", ""),
    "x(t) = A[1 − (1 + ωₙt)e^(−ωₙt)].": (
        "", r"x(t)=A\left[1-(1+\omega_n t)e^{-\omega_n t}\right]", ""),
    "Para ζ > 1, r₁,₂ = −ωₙ(ζ ∓ √(ζ²−1)):": (
        "Para ζ > 1, con raíces:", r"r_{1,2}=-\omega_n\left(\zeta\mp\sqrt{\zeta^2-1}\right)", ""),
    "x(t) = A[1 − (r₂e^(r₁t) − r₁e^(r₂t))/(r₂−r₁)].": (
        "", r"x(t)=A\left[1-\frac{r_2e^{r_1 t}-r_1e^{r_2 t}}{r_2-r_1}\right]", ""),
    "• ωₙ = 0,4 → x″ + 0,8x′ + 0,16x = 0,16.": (
        "• ωₙ = 0,4 →", r"x''+0{,}8x'+0{,}16x=0{,}16", ""),
    "• ωₙ = 0,8 → x″ + 1,6x′ + 0,64x = 0,64.": (
        "• ωₙ = 0,8 →", r"x''+1{,}6x'+0{,}64x=0{,}64", ""),
    "• ωₙ = 1,2 → x″ + 2,4x′ + 1,44x = 1,44.": (
        "• ωₙ = 1,2 →", r"x''+2{,}4x'+1{,}44x=1{,}44", ""),
    "Cada ecuación característica es (r + ωₙ)² = 0: amortiguamiento crítico.": (
        "Cada ecuación característica es:", r"(r+\omega_n)^2=0", "Amortiguamiento crítico."),
    "x(t) = 1 − (1 + ωₙt)e^(−ωₙt).": (
        "", r"x(t)=1-(1+\omega_n t)e^{-\omega_n t}", ""),
    "• ωₙ = 0,4 → x(t) = 1 − (1 + 0,4t)e^(−0,4t).": (
        "• ωₙ = 0,4 →", r"x(t)=1-(1+0{,}4t)e^{-0{,}4t}", ""),
    "• ωₙ = 0,8 → x(t) = 1 − (1 + 0,8t)e^(−0,8t).": (
        "• ωₙ = 0,8 →", r"x(t)=1-(1+0{,}8t)e^{-0{,}8t}", ""),
    "• ωₙ = 1,2 → x(t) = 1 − (1 + 1,2t)e^(−1,2t).": (
        "• ωₙ = 1,2 →", r"x(t)=1-(1+1{,}2t)e^{-1{,}2t}", ""),
    "x″ + 1,6x′ + 0,64x = 0,64A.": (
        "", r"x''+1{,}6x'+0{,}64x=0{,}64A", ""),
    "• Si A = 0,6 → x″ + 1,6x′ + 0,64x = 0,384.": (
        "• Si A = 0,6 →", r"x''+1{,}6x'+0{,}64x=0{,}384", ""),
    "• Si A = 1,0 → x″ + 1,6x′ + 0,64x = 0,640.": (
        "• Si A = 1,0 →", r"x''+1{,}6x'+0{,}64x=0{,}640", ""),
    "La ecuación característica homogénea es (r + 0,8)² = 0.": (
        "La ecuación característica homogénea es:", r"(r+0{,}8)^2=0", ""),
    "x(t) = A[1 − (1 + 0,8t)e^(−0,8t)].": (
        "", r"x(t)=A\left[1-(1+0{,}8t)e^{-0{,}8t}\right]", ""),
    "• Si A = 0,6 → x(t) = 0,6[1 − (1 + 0,8t)e^(−0,8t)].": (
        "• Si A = 0,6 →", r"x(t)=0{,}6\left[1-(1+0{,}8t)e^{-0{,}8t}\right]", ""),
    "• Si A = 1,0 → x(t) = 1 − (1 + 0,8t)e^(−0,8t).": (
        "• Si A = 1,0 →", r"x(t)=1-(1+0{,}8t)e^{-0{,}8t}", ""),
}


# Método para generar imágenes de las fórmulas
def formula_png(latex: str, dpi: int = 70) -> bytes:
    """Renderiza MathText con fracciones verticales y exponentes auténticos."""
    return _formula_png_cache(latex, dpi)


from functools import lru_cache

# Método para almacenar fórmulas renderizadas en caché
@lru_cache(maxsize=96)
def _formula_png_cache(latex: str, dpi: int) -> bytes:
    from io import BytesIO
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties

    # Tamaño de las ecuaciones
    TAMANO_FORMULA = 18
    stream = BytesIO()
    math_to_image(
        f"${latex}$",
        stream,
        prop=FontProperties(size=TAMANO_FORMULA),
        dpi=dpi,
        format="png",
        color=INK
    )
    return stream.getvalue()

# Método para abrir la carpeta de resultados
def abrir_carpeta(path):
    """Abre la carpeta en el explorador de archivos del sistema operativo."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# Método para encontrar el logo lateral
def buscar_logo():
    """Encuentra logo.png junto al programa, incluso en carpetas con otras mayúsculas."""
    candidatos = [
        BASE / "Images" / "logo.png", BASE / "Imagen" / "logo.png",
        BASE / "imagen" / "logo.png", BASE / "imagenes" / "logo.png",
        BASE / "Imágenes" / "logo.png", BASE / "images" / "logo.png",
        BASE / "logo.png",
    ]
    for ruta in candidatos:
        if ruta.is_file():
            return ruta
    # Buscar logo sin importar mayúsculas
    for carpeta in BASE.iterdir():
        if carpeta.is_dir() and carpeta.name.casefold() in {
            "images", "imagen", "imagenes", "imágenes"
        }:
            for archivo in carpeta.iterdir():
                if archivo.is_file() and archivo.name.casefold() == "logo.png":
                    return archivo
    return None


# Método para maximizar la aplicación
def ventana_maximizada(ventana):
    """Maximización normal (mantiene barra de título y acceso al sistema)."""
    if sys.platform.startswith("win"):
        try:
            ventana.state("zoomed")
            return
        except Exception:
            pass
    try:
        ventana.attributes("-zoomed", True)
        return
    except Exception:
        pass
    ventana.geometry(f"{ventana.winfo_screenwidth()}x{ventana.winfo_screenheight()}+0+0")


# Método para iniciar la interfaz gráfica
def lanzar_interfaz(datos):
    import tkinter as tk
    from tkinter import messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Clase principal de la interfaz gráfica
    class Aplicacion:
        # Método para configurar la ventana principal
        def __init__(self):
            # Configurar la ventana y el ícono de la aplicación
            self.root = tk.Tk()
            self.root.title("Reto Nº 3 IDS/IPS Simulación diferencial")
            # Ícono de la ventana desde Images/logo-app.png
            # Logo lateral independiente en Images/logo.png
            self.icono_aplicacion = None
            ruta_icono_app = BASE / "Images" / "logo-app.png"
            if ruta_icono_app.is_file():
                try:
                    self.icono_aplicacion = tk.PhotoImage(file=str(ruta_icono_app))
                    self.root.iconphoto(True, self.icono_aplicacion)
                except tk.TclError:
                    self.icono_aplicacion = None
            self.root.geometry("1230x840")  # Tamaño inicial de respaldo
            self.root.minsize(940, 640)
            self.root.configure(bg=BG)
            self.datos = datos
            self.nav_botones = {}
            self.pagina = None
            self.figura_widget = None
            self.logo_imagen = None  # Mantener la referencia del logo
            self.modo_pantalla_completa = False
            self._resize_id = None

            # Crear el menú lateral de navegación
            sidebar = tk.Frame(self.root, bg=INK, width=245)
            sidebar.pack(side="left", fill="y")
            sidebar.pack_propagate(False)
            # Logo junto al texto PUCE
            cabecera_logo = tk.Frame(sidebar, bg=INK)
            cabecera_logo.pack(anchor="w", padx=20, pady=(21, 9))
            ruta_logo = buscar_logo()
            if ruta_logo:
                try:
                    # Escalado del logo con alternativa de Tkinter
                    try:
                        from PIL import Image, ImageTk
                        with Image.open(ruta_logo) as img:
                            img = img.convert("RGBA")
                            img.thumbnail((54, 54), Image.Resampling.LANCZOS)
                            self.logo_imagen = ImageTk.PhotoImage(img)
                    except ImportError:
                        original = tk.PhotoImage(file=str(ruta_logo))
                        factor = max(1, math.ceil(max(original.width(), original.height()) / 54))
                        self.logo_imagen = original.subsample(factor, factor)
                    tk.Label(cabecera_logo, image=self.logo_imagen, bg=INK,
                             borderwidth=0).pack(side="left", padx=(0, 10))
                except Exception as exc:
                    print(f"Aviso: no fue posible mostrar el logo {ruta_logo}: {exc}", file=sys.stderr)
            else:
                print("Logo opcional: coloca el archivo Images/logo.png junto al script.", file=sys.stderr)
            tk.Label(cabecera_logo, text="PUCE", fg="#81CED4", bg=INK,
                     font=("Segoe UI", 24, "bold")).pack(side="left", anchor="center")
            tk.Label(sidebar, text="RETO N°3", fg=WHITE, bg=INK,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=22)
            tk.Label(sidebar, text="EDO de segundo orden\nSimulación IDS/IPS", fg="#B9C6D3",
                     bg=INK, justify="left", font=("Segoe UI", 10)).pack(anchor="w", padx=22, pady=(5, 30))

            for clave, texto in [
                ("amortiguamiento", "Amortiguamiento"),
                ("rapidez", "Rapidez"),
                ("demanda", "Demanda")
            ]:
                b = tk.Button(sidebar, text=texto, anchor="w", relief="flat", bd=0,
                              bg=INK, fg="#D4DEE7", activeforeground=WHITE,
                              cursor="hand2", font=("Segoe UI", 11, "bold"),
                              padx=20, pady=15, command=lambda c=clave: self.ir(c))
                b.pack(fill="x", padx=10, pady=4)
                self.nav_botones[clave] = b
            tk.Frame(sidebar, bg="#34495D", height=1).pack(fill="x", padx=20, pady=(22, 15))
            tk.Button(sidebar, text="Exportar recursos", command=self.exportar_otra_vez,
                      bg=TEAL, fg=WHITE, activebackground="#046577", activeforeground=WHITE,
                      borderwidth=0, padx=10, pady=12, cursor="hand2",
                      font=("Segoe UI", 10, "bold"), wraplength=190).pack(fill="x", padx=16, pady=5)
            tk.Button(sidebar, text="Abrir carpeta de resultados", command=self.abrir,
                      bg="#293F54", fg=WHITE, activebackground="#405668", activeforeground=WHITE,
                      borderwidth=0, padx=10, pady=12, cursor="hand2",
                      font=("Segoe UI", 10)).pack(fill="x", padx=16, pady=5)

            # Crear el área principal y el encabezado
            central = tk.Frame(self.root, bg=BG)
            central.pack(side="right", fill="both", expand=True)
            superior = tk.Frame(central, bg=WHITE, height=78, highlightbackground=LINE, highlightthickness=1)
            superior.pack(side="top", fill="x")
            superior.pack_propagate(False)
            self.encabezado = tk.Label(superior, text="", bg=WHITE, fg=INK,
                                      font=("Segoe UI", 19, "bold"), anchor="w")
            self.encabezado.pack(fill="both", expand=True, padx=30)

            # Crear el área desplazable
            area = tk.Frame(central, bg=BG)
            area.pack(fill="both", expand=True)
            self.scroll = tk.Canvas(area, bg=BG, highlightthickness=0, bd=0)
            barra = ttk.Scrollbar(area, orient="vertical", command=self.scroll.yview)
            self.scroll.configure(yscrollcommand=barra.set)
            barra.pack(side="right", fill="y")
            self.scroll.pack(side="left", fill="both", expand=True)
            self.contenido = tk.Frame(self.scroll, bg=BG)
            self.item = self.scroll.create_window((0, 0), window=self.contenido, anchor="nw")
            self.contenido.bind("<Configure>", lambda _e: self.scroll.configure(
                scrollregion=self.scroll.bbox("all")))
            self.scroll.bind("<Configure>", lambda e: self.scroll.itemconfigure(self.item, width=e.width))
            # Desplazamiento dentro del área principal
            self.scroll.bind("<Enter>", lambda _e: self.root.bind_all("<MouseWheel>", self.rueda))
            self.scroll.bind("<Leave>", lambda _e: self.root.unbind_all("<MouseWheel>"))
            self.root.bind("<Control-e>", lambda _e: self.exportar_otra_vez())
            # Atajos de pantalla completa
            self.root.bind("<F11>", self.alternar_pantalla_completa)
            self.root.bind("<Escape>", self.salir_pantalla_completa)
            self.root.bind("<Configure>", self.programar_ajuste_grafico)
            # Maximizar la ventana al iniciar
            self.root.after(80, lambda: ventana_maximizada(self.root))
            self.ir("amortiguamiento")

        # Método para activar o desactivar pantalla completa
        def alternar_pantalla_completa(self, _evento=None):
            self.modo_pantalla_completa = not self.modo_pantalla_completa
            self.root.attributes("-fullscreen", self.modo_pantalla_completa)
            if not self.modo_pantalla_completa:
                self.root.after(40, lambda: ventana_maximizada(self.root))
            return "break"

        # Método para salir de pantalla completa
        def salir_pantalla_completa(self, _evento=None):
            if self.modo_pantalla_completa:
                self.modo_pantalla_completa = False
                self.root.attributes("-fullscreen", False)
                self.root.after(40, lambda: ventana_maximizada(self.root))
            return "break"

        # Método para programar el ajuste del gráfico
        def programar_ajuste_grafico(self, evento):
            # Ajustar solo al cambiar la ventana principal
            if evento.widget is not self.root:
                return
            if self._resize_id is not None:
                self.root.after_cancel(self._resize_id)
            self._resize_id = self.root.after(120, self.ajustar_grafico)

        # Método para adaptar la altura del gráfico
        def ajustar_grafico(self):
            self._resize_id = None
            if self.figura_widget is None:
                return
            widget = self.figura_widget.get_tk_widget()
            if not widget.winfo_exists():
                return
            # Altura del gráfico según la ventana
            altura = max(370, self.root.winfo_height() - 250)
            if int(widget.cget("height")) != altura:
                widget.configure(height=altura)

        # Método para mostrar el gráfico a pantalla completa
        def ver_grafico_completo(self, clave):
            """Visor exclusivo del gráfico a pantalla completa, con Esc para salir."""
            visor = tk.Toplevel(self.root)
            visor.title("Gráfico a pantalla completa · " + TITULOS[clave])
            visor.configure(bg=WHITE)
            barra = tk.Frame(visor, bg=INK)
            barra.pack(side="top", fill="x")
            tk.Label(barra, text=TITULOS[clave], fg=WHITE, bg=INK,
                     font=("Segoe UI", 12, "bold")).pack(side="left", padx=18, pady=10)
            tk.Button(barra, text="Cerrar gráfico  (Esc)", command=visor.destroy,
                      bg=TEAL, fg=WHITE, relief="flat", padx=16, pady=8,
                      cursor="hand2").pack(side="right", padx=12, pady=6)
            # Generar el gráfico de la sección
            fig = construir_figura(clave, self.datos[clave])
            lienzo = FigureCanvasTkAgg(fig, master=visor)
            zona = lienzo.get_tk_widget()
            zona.configure(bg=WHITE, highlightthickness=0)
            zona.pack(side="top", fill="both", expand=True, padx=14, pady=12)
            lienzo.draw()
            visor.bind("<Escape>", lambda _e: visor.destroy())
            visor.after(35, lambda: visor.attributes("-fullscreen", True) if visor.winfo_exists() else None)
            visor.focus_set()

        # Método para desplazar el contenido con el ratón
        def rueda(self, event):
            self.scroll.yview_scroll(int(-event.delta / 120), "units")

        # Método para abrir los archivos exportados
        def abrir(self):
            try:
                abrir_carpeta(SALIDA)
            except Exception as exc:
                messagebox.showerror("No se pudo abrir", str(exc))

        # Método para exportar desde el botón
        def exportar_otra_vez(self):
            try:
                rutas = exportar(self.datos)
                messagebox.showinfo("Exportación completada", "Se generaron:\n\n" +
                                    "\n".join(p.name for p in rutas) +
                                    f"\n\nCarpeta: {SALIDA}")
            except Exception as exc:
                messagebox.showerror("Error al exportar", str(exc))

        # Método para crear una tarjeta explicativa
        def tarjeta(self, titulo, texto, ancho=None):
            frame = tk.Frame(self.contenido, bg=WHITE, padx=22, pady=16,
                             highlightthickness=1, highlightbackground=LINE)
            frame.pack(fill="x", padx=26, pady=8)
            tk.Label(frame, text=titulo, bg=WHITE, fg=TEAL, anchor="w",
                     font=("Segoe UI", 11, "bold")).pack(fill="x", pady=(0, 8))

            # Conservar el estilo del texto
            # Formatear las ecuaciones con MathText
            # Mostrar fracciones mediante imágenes matemáticas
            etiquetas = []
            # Método para mostrar texto explicativo
            def texto_plano(mensaje):
                if not mensaje:
                    return
                etiqueta = tk.Label(frame, text=mensaje, bg=WHITE, fg=INK,
                                    justify="left", anchor="w", font=("Segoe UI", 10),
                                    wraplength=790)
                etiqueta.pack(fill="x", pady=(1, 1))
                etiquetas.append(etiqueta)

            # Método para mostrar fórmulas formateadas
            def formula_visual(latex):
                from io import BytesIO
                from PIL import Image, ImageTk  # Cargar imágenes matemáticas con Pillow
                contenido_png = formula_png(latex)
                imagen = Image.open(BytesIO(contenido_png)).convert("RGBA")
                # Ajustar las fórmulas al ancho disponible
                ancho_disponible = max(400, self.root.winfo_screenwidth() - 385)
                if imagen.width > ancho_disponible:
                    factor = ancho_disponible / imagen.width
                    imagen = imagen.resize(
                        (ancho_disponible, max(1, round(imagen.height*factor))),
                        Image.Resampling.LANCZOS
                    )
                photo = ImageTk.PhotoImage(imagen)
                etiqueta = tk.Label(frame, image=photo, bg=WHITE, bd=0, anchor="w")
                etiqueta.image = photo  # Conservar la imagen de la fórmula
                etiqueta.pack(anchor="w", pady=(2, 5))

            # Mostrar texto o fórmula según cada línea
            for linea in texto.splitlines():
                # Dibujar los nuevos pasos con exponentes y fracciones reales
                if linea.startswith("@FORMULA:"):
                    formula_visual(linea[len("@FORMULA:"):])
                    continue
                formula = MATEMATICA_VISUAL.get(linea.strip())
                if formula is None:
                    texto_plano(linea)
                else:
                    antes, latex, despues = formula
                    texto_plano(antes)
                    formula_visual(latex)
                    texto_plano(despues)

            frame.bind("<Configure>", lambda e, labels=etiquetas: [
                lab.configure(wraplength=max(e.width-44, 300)) for lab in labels
            ])
            return frame

        # Método para navegar entre las tres secciones
        def ir(self, clave):
            if clave == self.pagina:
                self.scroll.yview_moveto(0)
                return
            self.pagina = clave
            for key, button in self.nav_botones.items():
                button.configure(bg=TEAL if key == clave else INK,
                                 fg=WHITE if key == clave else "#D4DEE7")
            self.encabezado.configure(text={
                "amortiguamiento": "Comparación de amortiguamiento",
                "rapidez": "Rapidez natural de respuesta",
                "demanda": "Nivel de demanda del ataque"
            }[clave])
            # Limpiar el contenido de la página anterior
            for widget in self.contenido.winfo_children():
                widget.destroy()
            self.figura_widget = None

            descripcion = {
                "amortiguamiento": "Se mantiene A = 1 y ωₙ = 0,8 min⁻¹; varía ζ entre 0,35, 1,0 y 1,8.",
                "rapidez": "Se mantiene ζ = 1 y A = 1; varía ωₙ entre 0,4, 0,8 y 1,2 min⁻¹.",
                "demanda": "Se mantiene ζ = 1 y ωₙ = 0,8 min⁻¹; varía A entre 0,6 y 1,0."
            }[clave]
            tk.Label(self.contenido, text=descripcion, bg=BG, fg=MUTED,
                     font=("Segoe UI", 11), anchor="w").pack(fill="x", padx=28, pady=(20, 8))
            panel = tk.Frame(self.contenido, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
            panel.pack(fill="x", padx=26, pady=8)
            herramientas = tk.Frame(panel, bg=WHITE)
            herramientas.pack(fill="x", padx=16, pady=(12, 0))
            tk.Label(herramientas, text="GRÁFICO INTERACTIVO", bg=WHITE, fg=TEAL,
                     font=("Segoe UI", 10, "bold")).pack(side="left")
            tk.Button(herramientas, text="⛶", bg=INK, fg=WHITE,
                      activebackground=TEAL, activeforeground=WHITE,
                      relief="flat", borderwidth=0, cursor="hand2",
                      padx=12, pady=8, font=("Segoe UI", 10, "bold"),
                      command=lambda c=clave: self.ver_grafico_completo(c)).pack(side="right")
            # Generar el gráfico de la sección
            fig = construir_figura(clave, self.datos[clave])
            # Adaptar el gráfico al tamaño disponible
            lienzo = FigureCanvasTkAgg(fig, master=panel)
            self.figura_widget = lienzo
            widget = lienzo.get_tk_widget()
            widget.configure(height=max(370, self.root.winfo_height()-250),
                             bg=WHITE, highlightthickness=0)
            widget.pack(fill="both", expand=True, padx=12, pady=(5, 12))
            lienzo.draw()

            # Mostrar las métricas antes del desarrollo matemático
            self.tabla(clave)
            etiqueta = tk.Label(self.contenido, text="DESARROLLO MATEMÁTICO",
                                bg=BG, fg=INK, anchor="w", font=("Segoe UI", 12, "bold"))
            etiqueta.pack(fill="x", padx=28, pady=(22, 4))
            # Mostrar los pasos del desarrollo matemático
            for nombre, explicacion in pasos_de_resolucion(clave, self.datos[clave]):
                self.tarjeta(nombre, explicacion)
            footer = ("RESULTADO INTERPRETADO: " + {
                "amortiguamiento": "Los tres sistemas son estables, pero sus transitorios son distintos.",
                "rapidez": "La respuesta es más rápida al elevar ωₙ en el modelo supuesto.",
                "demanda": "La amplitud cambia con A; el tiempo relativo al 95 % permanece igual."
            }[clave])
            tk.Label(self.contenido, text=footer, bg=BG, fg=TEAL, font=("Segoe UI", 10, "bold"),
                     anchor="w", wraplength=780).pack(fill="x", padx=28, pady=(12, 30))
            self.scroll.yview_moveto(0)

        # Método para mostrar los resultados numéricos
        def tabla(self, clave):
            tarjeta = tk.Frame(self.contenido, bg=WHITE, padx=18, pady=15,
                               highlightbackground=LINE, highlightthickness=1)
            tarjeta.pack(fill="x", padx=26, pady=8)
            tk.Label(tarjeta, text="RESULTADOS NUMÉRICOS DE LA SIMULACIÓN", bg=WHITE,
                     fg=TEAL, font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x", pady=(0, 10))
            nombres = ("Caso", "Pico", "Primer 95 %", "Estabilización ±5 %", "Sobreimpulso")
            style = ttk.Style(self.root)
            style.configure("Reto.Treeview", rowheight=30, font=("Segoe UI", 9))
            style.configure("Reto.Treeview.Heading", font=("Segoe UI", 9, "bold"))
            # Crear tabla de métricas por escenario
            tree = ttk.Treeview(tarjeta, columns=nombres, show="headings", height=len(self.datos[clave]),
                                style="Reto.Treeview")
            for nom in nombres:
                tree.heading(nom, text=nom)
                tree.column(nom, width=180 if nom == "Caso" else 130,
                            minwidth=85, anchor="w" if nom == "Caso" else "center", stretch=True)
            for fila in self.datos[clave]:
                m = fila["metricas"]
                tree.insert("", "end", values=(
                    fila["etiqueta"], fnum(m["pico"], 3),
                    fnum(m["t95_min"]) + " min",
                    fnum(m["asentamiento_5pct_min"]) + " min",
                    fnum(m["sobreimpulso_pct"], 1) + " %"))
            tree.pack(fill="x")
            tk.Label(tarjeta, text="Primer 95 % = primer cruce de 0,95·A; estabilización = permanece en A ±5 %.",
                     bg=WHITE, fg=MUTED, font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(8, 0))

        # Método para mantener abierta la aplicación
        def ejecutar(self):
            self.root.mainloop()

    Aplicacion().ejecutar()


# Método principal de la aplicación
def main():
    try:
        # Calcular datos sin exportar al iniciar
        datos = calcular_datos()
        print("El programa se ejecutó de forma correcta")
        lanzar_interfaz(datos)
        print("El programa se ha cerrado con éxito")
        return 0
    except Exception as exc:
        import tkinter as tk
        from tkinter import messagebox
        ventana = tk.Tk()
        ventana.withdraw()
        messagebox.showerror("Error de la aplicación", f"No se pudo iniciar la aplicación:\n\n{exc}")
        ventana.destroy()
        return 1
    except KeyboardInterrupt:
            print("\nPrograma finalizado por el usuario.")


if __name__ == "__main__":
    raise SystemExit(main())
