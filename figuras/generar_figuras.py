"""Genera las figuras del README a partir de los datos del repositorio.

Uso, desde la raíz del proyecto:

    python figuras/generar_figuras.py

- factores_{claro,oscuro}.png: tasa de reingreso en menos de 30 días por factor, sobre los
  101.766 ingresos originales (las mismas tablas que el notebook 03).
- diagnostico_especialidad_{claro,oscuro}.png: por capítulo del diagnóstico principal y por
  especialidad, sobre train.csv + test.csv (el apartado 3.5 del notebook 04).

Cada figura se guarda en versión clara y oscura: GitHub muestra la que corresponde al tema del lector.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = Path(__file__).resolve().parent

# Colores validados para daltonismo y contraste sobre los fondos de GitHub (claro #ffffff, oscuro #0d1117)
TEMAS = {
    'claro': {'fondo': '#ffffff', 'texto': '#0b0b0b', 'texto_2': '#52514e', 'linea': '#d8d7d0',
              'encima': '#e34948', 'media': '#a9a8a2', 'debajo': '#2a78d6'},
    'oscuro': {'fondo': '#0d1117', 'texto': '#ffffff', 'texto_2': '#c3c2b7', 'linea': '#383835',
               'encima': '#e66767', 'media': '#6e6d68', 'debajo': '#3987e5'},
}
MARGEN_NEUTRO = 0.5   # puntos: a menos distancia de la media, la categoría se pinta en gris

# Geometría en pulgadas: 10 de ancho para que el texto se lea bien a la anchura del README
ANCHO, IZQ, DER, HUECO_COL = 10.0, 0.25, 0.25, 0.40
FILA_BARRA, GROSOR_BARRA, RADIO = 0.40, 0.17, 0.035
T_CATEGORIA, T_CASOS, T_VALOR = 9.5, 7.5, 9.5    # tamaños de letra (pt)


def pct(x):
    return f'{x:.1f}'.replace('.', ',') + ' %'


def miles(n):
    return f'{int(n):,}'.replace(',', '.')


def tasas(es_30, grupo, orden=None, nombres=None):
    """Tasa de <30 (%) y número de ingresos de cada categoría de `grupo`."""
    t = pd.DataFrame({'tasa': es_30.groupby(grupo).mean() * 100, 'n': grupo.value_counts()})
    t = t.loc[orden] if orden is not None else t.sort_values('tasa', ascending=False)
    return t.rename(index=nombres or {})


# ======================================================================= datos
def datos_factores():
    """Factores del notebook 03, sobre el dataset original (101.766 ingresos, sin exclusiones)."""
    df = pd.read_csv(RAIZ / 'data' / 'raw' / 'diabetic_data.csv', low_memory=False)
    df[['max_glu_serum', 'A1Cresult']] = df[['max_glu_serum', 'A1Cresult']].fillna('Not_Measured')
    es_30 = df['readmitted'] == '<30'

    ingresos_previos = df['number_inpatient'].clip(upper=4).map(
        {0: 'Ninguno', 1: 'Uno', 2: 'Dos', 3: 'Tres', 4: 'Cuatro o más'})
    destinos = {1: 'A casa', 3: 'Residencia asistida', 6: 'Atención domiciliaria', 2: 'Otro hospital',
                22: 'Rehabilitación', 5: 'Otra institución', 7: 'Alta voluntaria'}
    destino = df['discharge_disposition_id'].map(destinos)          # los 7 destinos con más de 600 ingresos
    metformina = df['metformin'].replace({'Up': 'Ajustada', 'Down': 'Ajustada'})

    paneles = [
        ('Fragilidad', 'Ingresos hospitalarios en el año previo',
         tasas(es_30, ingresos_previos, ['Ninguno', 'Uno', 'Dos', 'Tres', 'Cuatro o más'])),
        ('Fragilidad', 'Destino al alta', tasas(es_30[destino.notna()], destino.dropna())),
        ('Inestabilidad del tratamiento', 'Insulina',
         tasas(es_30, df['insulin'], ['No', 'Steady', 'Up', 'Down'],
               {'No': 'No la toma', 'Steady': 'Dosis estable', 'Up': 'Dosis subida', 'Down': 'Dosis bajada'})),
        ('Inestabilidad del tratamiento', 'Glucemia sérica',
         tasas(es_30, df['max_glu_serum'], ['Not_Measured', 'Norm', '>200', '>300'],
               {'Not_Measured': 'No medida', 'Norm': 'Normal', '>200': 'Más de 200 mg/dL', '>300': 'Más de 300 mg/dL'})),
        ('Seguimiento', 'Hemoglobina glicosilada (HbA1c)',
         tasas(es_30, df['A1Cresult'], ['Not_Measured', 'Norm', '>7', '>8'],
               {'Not_Measured': 'No medida', 'Norm': 'Normal', '>7': 'Más del 7 %', '>8': 'Más del 8 %'})),
        ('Seguimiento', 'Metformina',
         tasas(es_30, metformina, ['No', 'Steady', 'Ajustada'],
               {'No': 'No la toma', 'Steady': 'Dosis estable', 'Ajustada': 'Dosis ajustada'})),
    ]
    return paneles, es_30.mean() * 100, len(df)


def datos_diagnostico_especialidad():
    """Apartado 3.5 del notebook 04: train + test (99.343 ingresos, sin fallecidos ni hospice)."""
    df = pd.concat([pd.read_csv(RAIZ / 'data' / 'processed' / f) for f in ('train.csv', 'test.csv')])
    es_30 = df['readmitted_30d'] == 1

    def por_dummies(prefijo, nombres=None, n_min=0):
        filas = {}
        for col in [c for c in df.columns if c.startswith(prefijo)]:
            marca = df[col] == 1
            if marca.sum() >= n_min:
                filas[col.removeprefix(prefijo)] = {'tasa': es_30[marca].mean() * 100, 'n': marca.sum()}
        t = pd.DataFrame(filas).T.sort_values('tasa', ascending=False)
        return t.rename(index=nombres or {})

    especialidades = {'Unknown': 'Sin registrar', 'InternalMedicine': 'Medicina interna',
                      'Emergency/Trauma': 'Urgencias', 'Family/GeneralPractice': 'Medicina familiar',
                      'Cardiology': 'Cardiología', 'Surgery-General': 'Cirugía general',
                      'Nephrology': 'Nefrología', 'Orthopedics': 'Traumatología',
                      'Orthopedics-Reconstructive': 'Traumatología reconstructiva', 'Radiologist': 'Radiología'}
    diagnostico = por_dummies('diag_1_').drop(index='Unknown')          # 20 ingresos: no se interpreta
    paneles = [
        (None, 'Capítulo del diagnóstico principal', diagnostico),
        (None, 'Especialidad (más de 1.000 ingresos)', por_dummies('medical_specialty_', especialidades, 1000)),
    ]
    return paneles, es_30.mean() * 100, len(df)


# ===================================================================== dibujo
def ancho_texto(fig, texto, **estilo):
    """Ancho en pulgadas de un texto ya renderizado con ese estilo."""
    t = fig.text(0, 0, texto, **estilo)
    ancho = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return ancho


def dibujar(paneles, media, titulo, subtitulo, fuente, tema, archivo):
    c = TEMAS[tema]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    filas = [paneles[i:i + 2] for i in range(0, len(paneles), 2)]
    altos = [max(len(p[2]) for p in fila) * FILA_BARRA for fila in filas]
    cabecera, pie = 1.45, 0.45
    grupo_alto, titulo_alto, hueco_fila = 0.30, 0.42, 0.30
    con_grupo = any(p[0] for p in paneles)
    alto_total = (cabecera + pie + sum(altos)
                  + len(filas) * (titulo_alto + (grupo_alto if con_grupo else 0)) + (len(filas) - 1) * hueco_fila)

    fig = plt.figure(figsize=(ANCHO, alto_total), dpi=200, facecolor=c['fondo'])
    ancho_panel = (ANCHO - IZQ - DER - HUECO_COL) / 2

    # Reparto horizontal de cada panel, medido sobre los textos reales: nombres | barras | porcentaje
    nombres = max(max(ancho_texto(fig, str(k), fontsize=T_CATEGORIA) for k in t.index) for _, _, t in paneles)
    nombres = max(nombres, max(ancho_texto(fig, f'{miles(v)} ingresos', fontsize=T_CASOS)
                               for _, _, t in paneles for v in t['n'])) + 0.12
    valor = max(ancho_texto(fig, pct(v), fontsize=T_VALOR, fontweight='bold')
                for _, _, t in paneles for v in t['tasa']) + 0.14
    barras = ancho_panel - nombres - valor
    tasa_max = max(t['tasa'].max() for _, _, t in paneles)
    dx = tasa_max / barras                                  # unidades de datos (puntos %) por pulgada
    x_min, x_max = -nombres * dx, tasa_max + valor * dx

    def y_fig(pulgadas_desde_arriba):
        return 1 - pulgadas_desde_arriba / alto_total

    # Cabecera: título, subtítulo y leyenda
    fig.text(IZQ / ANCHO, y_fig(0.30), titulo, fontsize=15, fontweight='bold', color=c['texto'], va='top')
    fig.text(IZQ / ANCHO, y_fig(0.66), subtitulo, fontsize=10, color=c['texto_2'], va='top', linespacing=1.45)
    leyenda = [('encima', 'Por encima de la media'),
               ('media', f'A menos de {pct(MARGEN_NEUTRO)[:-2]} puntos de la media'),
               ('debajo', 'Por debajo de la media')]
    x_ley, y_ley = IZQ, y_fig(1.24)
    for clave, texto in leyenda:
        fig.patches.append(FancyBboxPatch((x_ley / ANCHO, y_ley - 0.065 / alto_total), 0.13 / ANCHO, 0.13 / alto_total,
                                          boxstyle='round,pad=0,rounding_size=0.004', transform=fig.transFigure,
                                          mutation_aspect=ANCHO / alto_total, color=c[clave], lw=0))
        fig.text((x_ley + 0.19) / ANCHO, y_ley, texto, fontsize=9, color=c['texto_2'], va='center_baseline')
        x_ley += 0.19 + ancho_texto(fig, texto, fontsize=9) + 0.40

    arriba = cabecera
    for fila, alto in zip(filas, altos):
        if con_grupo:
            fig.text(IZQ / ANCHO, y_fig(arriba + 0.20), fila[0][0].upper(), fontsize=8.5, fontweight='bold',
                     color=c['texto_2'], va='center')
            arriba += grupo_alto
        for i_col, (_, nombre_panel, t) in enumerate(fila):
            x0 = IZQ + i_col * (ancho_panel + HUECO_COL)
            n = len(t)
            alto_panel = n * FILA_BARRA
            ax = fig.add_axes([x0 / ANCHO, y_fig(arriba + titulo_alto + alto_panel),
                               ancho_panel / ANCHO, alto_panel / alto_total])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(n - 0.5, -0.5)
            ax.axis('off')
            ax.text(0, 1 + 0.25 / alto_panel, nombre_panel, transform=ax.transAxes, fontsize=11,
                    fontweight='bold', color=c['texto'], va='bottom')

            dy = 1 / FILA_BARRA                             # unidades de datos (filas) por pulgada
            radio_x, h = RADIO * dx, GROSOR_BARRA * dy
            ax.plot([0, 0], [-0.5, n - 0.5], color=c['linea'], lw=1, solid_capstyle='butt')
            ax.plot([media, media], [-0.5, n - 0.5], color=c['texto_2'], lw=1.2, solid_capstyle='butt', zorder=2)
            ax.text(media, -0.5 - 0.03 * dy, f'media {pct(media)}', fontsize=8, color=c['texto_2'],
                    ha='center', va='bottom')

            for y, (categoria, fila_t) in enumerate(t.iterrows()):
                tasa, casos = fila_t['tasa'], fila_t['n']
                diferencia = tasa - media
                color = c['media'] if abs(diferencia) < MARGEN_NEUTRO else c['encima'] if diferencia > 0 else c['debajo']
                # Barra con el extremo de datos redondeado y la base recta
                ax.add_patch(FancyBboxPatch((0, y - h / 2), tasa, h, boxstyle=f'round,pad=0,rounding_size={radio_x}',
                                            mutation_aspect=dy / dx, color=color, lw=0, zorder=3))
                ax.add_patch(Rectangle((0, y - h / 2), min(tasa, 2 * radio_x), h, color=color, lw=0, zorder=3))
                # Nombre de la categoría y, debajo, su número de ingresos
                ax.text(-0.10 * dx, y + 0.02 * dy, categoria, ha='right', va='baseline',
                        fontsize=T_CATEGORIA, color=c['texto'])
                ax.text(-0.10 * dx, y + 0.06 * dy, f'{miles(casos)} ingresos', ha='right', va='top',
                        fontsize=T_CASOS, color=c['texto_2'])
                # Porcentaje en la punta. La línea de la media va detrás de las barras, y un fondo desde
                # la punta hasta el final del porcentaje evita que asome entre ambos o lo atraviese
                valor_txt = ax.text(tasa + 0.07 * dx, y, pct(tasa), ha='left', va='center', fontsize=T_VALOR,
                                    fontweight='bold', color=c['texto'], zorder=5)
                borde = valor_txt.get_window_extent(renderer=fig.canvas.get_renderer()).x1
                x_fin = ax.transData.inverted().transform((borde, 0))[0]
                ax.add_patch(Rectangle((tasa, y - h / 2), x_fin - tasa + 0.04 * dx, h, color=c['fondo'],
                                       lw=0, zorder=3.5))
        arriba += titulo_alto + alto + hueco_fila

    fig.text(IZQ / ANCHO, 0.22 / alto_total, fuente, fontsize=8, color=c['texto_2'], va='center')
    fig.savefig(SALIDA / archivo, facecolor=c['fondo'], metadata={'Software': None})
    plt.close(fig)


def main():
    fuente = 'Fuente: Diabetes 130-US hospitals (UCI). Figura generada con figuras/generar_figuras.py'

    paneles, media, n = datos_factores()
    for tema in TEMAS:
        dibujar(paneles, media,
                '¿Qué anticipa un reingreso en menos de 30 días?',
                f'Porcentaje de ingresos seguidos de un reingreso en menos de 30 días, sobre {miles(n)} ingresos.\n'
                f'La línea marca la media de todos ellos ({pct(media)}); bajo cada categoría, su número de ingresos.',
                fuente, tema, f'factores_{tema}.png')
    tabla_factores = pd.concat({p[1]: p[2] for p in paneles})

    paneles, media, n = datos_diagnostico_especialidad()
    for tema in TEMAS:
        dibujar(paneles, media,
                'Diagnóstico principal y especialidad',
                f'Porcentaje de ingresos seguidos de un reingreso en menos de 30 días, sobre {miles(n)} ingresos\n'
                f'(sin fallecidos ni cuidados paliativos). La línea marca la media ({pct(media)}).',
                fuente, tema, f'diagnostico_especialidad_{tema}.png')
    tabla_diag = pd.concat({p[1]: p[2] for p in paneles})

    # Las cifras de las figuras, en tabla
    pd.set_option('display.width', 120)
    print(tabla_factores.round(2).to_string(), '\n')
    print(tabla_diag.round(2).to_string())


if __name__ == '__main__':
    main()
