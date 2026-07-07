import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
import pygwalker as pyg 
from io import BytesIO

# Estilo global para los gráficos de Matplotlib y Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 50, 'axes.titlesize': 14, 'axes.labelsize': 11})

# =========================================================
# 1. FUNCIÓN OPTIMIZADA DE CARGA Y PROCESAMIENTO DE DATOS
# =========================================================
@st.cache_data(show_spinner=False)
def procesar_datos_sies():
    # Nombres exactos de tus archivos locales
    PATH_MATRICULA = "MMatricula_2007_2025_WEB_15_07_2025.csv"
    PATH_OFERTA = "Oferta_Academica_2010_al_2025_SIES_02_06_2025_WEB_E.csv"
    
    # 1.1 Carga inteligente de Matrícula
    cols_df2 = [
        'AÑO', 'CÓDIGO CARRERA', 'NOMBRE CARRERA', 'ÁREA DEL CONOCIMIENTO', 
        'CINE-F 2013 SUBAREA', 'CLASIFICACIÓN INSTITUCIÓN NIVEL 1', 
        'CLASIFICACIÓN INSTITUCIÓN NIVEL 3', 'TOTAL MATRÍCULA', 
        'TOTAL MATRÍCULA MUJERES', 'TOTAL MATRÍCULA HOMBRES', 
        'TOTAL MATRÍCULA NO BINARIOS O INDEFINIDOS', 'MODALIDAD'
    ]
    
    # Forzamos 'CÓDIGO CARRERA' como texto (str) desde la lectura para evitar conflictos de mezcla
    df2 = pd.read_csv(
        PATH_MATRICULA, sep=';', encoding='latin-1', low_memory=False,
        on_bad_lines='skip', usecols=lambda c: c in cols_df2,
        dtype={'CÓDIGO CARRERA': str}
    )
    
    # Limpieza del atributo temporal Año
    df2['AÑO'] = df2['AÑO'].astype(str).str.replace('MAT_', '', case=False).str.strip()
    df2['AÑO'] = pd.to_numeric(df2['AÑO'], errors='coerce').fillna(0).astype(int)
    
    # Filtro temporal estricto (2017-2025)
    df2_limpio = df2[(df2['AÑO'] >= 2017) & (df2['AÑO'] <= 2025)].copy()
    
    # Imputación segura de nulos numéricos
    cols_num_df2 = ['TOTAL MATRÍCULA', 'TOTAL MATRÍCULA MUJERES', 'TOTAL MATRÍCULA HOMBRES', 'TOTAL MATRÍCULA NO BINARIOS O INDEFINIDOS']
    for col in cols_num_df2:
        if col in df2_limpio.columns:
            df2_limpio[col] = pd.to_numeric(df2_limpio[col], errors='coerce').fillna(0).astype(int)
            
    df2_limpio['MODALIDAD'] = df2_limpio['MODALIDAD'].fillna('PRESENCIAL')
    
    # 1.2 Carga inteligente de Oferta Académica / Aranceles
    cols_df3 = ['Año', 'Código Carrera', 'Arancel Anual', 'Matrícula Anual']
    
    # Forzamos 'Código Carrera' como texto (str) también aquí desde el nacimiento del DataFrame
    df3 = pd.read_csv(
        PATH_OFERTA, sep=';', encoding='latin-1', low_memory=False,
        on_bad_lines='skip', usecols=lambda c: c.strip() in cols_df3,
        dtype={'Código Carrera': str}
    )
    df3.columns = df3.columns.str.strip()
    
    df3['Año'] = df3['Año'].astype(str).str.replace('OFE_', '', case=False).str.strip()
    df3['Año'] = pd.to_numeric(df3['Año'], errors='coerce').fillna(0).astype(int)
    
    df3_limpio = df3[(df3['Año'] >= 2017) & (df3['Año'] <= 2025)].copy()
    
    # Limpieza financiera (Arancel y Matrícula Anual se mantienen numéricos)
    for col in ['Arancel Anual', 'Matrícula Anual']:
        df3_limpio[col] = df3_limpio[col].astype(str).str.replace('.', '', regex=False).str.strip()
        df3_limpio[col] = pd.to_numeric(df3_limpio[col], errors='coerce').fillna(0).astype(int)
        
    # 1.3 Normalización de llaves analíticas (Minúsculas, sin tildes ni espacios)
    def normalizar_columnas(columnas):
        return (columnas.str.strip().str.lower()
                        .str.replace('á', 'a').str.replace('é', 'e')
                        .str.replace('í', 'i').str.replace('ó', 'o').str.replace('ú', 'u'))
        
    df2_limpio.columns = normalizar_columnas(df2_limpio.columns)
    df3_limpio.columns = normalizar_columnas(df3_limpio.columns)
    
    # Asegurar limpieza extra en strings de unión (evita problemas de espacios ocultos)
    df2_limpio['codigo carrera'] = df2_limpio['codigo carrera'].astype(str).str.strip()
    df3_limpio['codigo carrera'] = df3_limpio['codigo carrera'].astype(str).str.strip()
    
    # Sincronizar el tipo de datos del año de forma estricta
    df2_limpio['año'] = df2_limpio['año'].astype(int)
    df3_limpio['año'] = df3_limpio['año'].astype(int)
    
    # Eliminación de registros duplicados en llaves compuestas
    df_aran_unico = df3_limpio.drop_duplicates(subset=['codigo carrera', 'año']).copy()
    df_aran_final = df_aran_unico[['codigo carrera', 'año', 'arancel anual', 'matricula anual']].copy()
    
    # Cruzamiento relacional indexado (Left Join) - Ahora ambos campos clave son de tipo str (objeto)
    df_consolidado = pd.merge(df2_limpio, df_aran_final, on=['codigo carrera', 'año'], how='left')
    return df_consolidado

# Ejecución de la carga con aviso dinámico en la UI
with st.spinner("Optimizando y cargando microdatos del SIES (2017-2025)... Espere un momento."):
    df = procesar_datos_sies()

# =========================================================
# 2. PANEL DE CONTROL LATERAL (SIDEBAR DE NAVEGACIÓN)
# =========================================================

st.sidebar.title("Módulos de Control")
st.sidebar.markdown("Use los botones para navegar de forma fluida por los distintos análisis de la defense.")

modulo = st.sidebar.radio(
    "Seleccione Visualización:",
    ["1. Análisis General de la Matrícula", "2. Tendencias de Macroáreas", "3. Enfoque Sector Económico", "4. Cluster Control de Gestión"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Nota Metodológica:** Los datos utilizados provienen de los registros oficiales de Matrículas y Oferta Académica publicados por el SIES (MINEDUC).")


# =========================================================
# 3. INTERFAZ GRÁFICA PRINCIPAL SEGÚN MÓDULO SELECCIONADO
# =========================================================

# --- MÓDULO 1: ANÁLISIS GENERAL ---
if modulo == "1. Análisis General de la Matrícula":
    st.header("Análisis General de la Matrícula en Educación Superior (2017–2025)")
    st.markdown("Evolución temporal, tasas de variación de los estudiantes activos y distribución por género a nivel país.")
    
    # Cálculos analíticos
    evolucion = df.groupby('año')['total matricula'].sum().reset_index()
    evolucion['VARIACIÓN ANUAL (%)'] = evolucion['total matricula'].pct_change() * 100
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("Evolución de Volumetría")
        st.dataframe(
            evolucion.style.format({'total matricula': '{:,.0f}', 'VARIACIÓN ANUAL (%)': '{:.2f}%'}),
            use_container_width=True
        )
        
        fig_var, ax_var = plt.subplots(figsize=(6, 3.5))
        sns.lineplot(data=evolucion, x='año', y='VARIACIÓN ANUAL (%)', marker='o', color='crimson', ax=ax_var)
        ax_var.set_title("Tasa de Variación Interanual (%)")
        st.pyplot(fig_var)
        
    with col2:
        st.subheader("Tendencia Global de Inscripciones")
        fig_line, ax_line = plt.subplots(figsize=(8, 5.3))
        sns.lineplot(data=evolucion, x='año', y='total matricula', marker='o', linewidth=2.5, color='darkblue', ax=ax_line)
        ax_line.set_title("Evolución Absoluta de Alumnos Matriculados")
        ax_line.set_ylabel("Cantidad de Estudiantes")
        st.pyplot(fig_line)
        
    st.markdown("---")
    st.subheader("Comportamiento de la Matrícula según Identidad de Género")
    
    ev_genero = df.groupby('año')[['total matricula mujeres', 'total matricula hombres', 'total matricula no binarios o indefinidos']].sum().reset_index()
    st.dataframe(ev_genero.style.format('{:,.0f}'), use_container_width=True)
    
    fig_gen, ax_gen = plt.subplots(figsize=(12, 4.5))
    for col_gen, label, color in [('total matricula mujeres', 'Mujeres', 'purple'), ('total matricula hombres', 'Hombres', 'teal'), ('total matricula no binarios o indefinidos', 'No Binarios / Indefinidos', 'gold')]:
        sns.lineplot(data=ev_genero, x='año', y=col_gen, marker='o', label=label, color=color, ax=ax_gen)
    ax_gen.set_title("Evolución Temporal de Segmentos de Género")
    ax_gen.set_ylabel("Estudiantes")
    ax_gen.legend(loc="upper left")
    st.pyplot(fig_gen)


    # --- MÓDULO 2: TENDENCIAS DE MACROÁREAS ---
elif modulo == "2. Tendencias de Macroáreas":
    st.header("Tendencias en Áreas del Conocimiento y Subáreas")
    st.markdown("Identificación de las macroáreas con mayor dinamismo y demanda de matrícula en Chile.")
    
    matriz_areas = df.groupby(['area del conocimiento', 'año'])['total matricula'].sum().unstack().fillna(0)
    matriz_areas['Crecimiento Absoluto'] = matriz_areas[2025] - matriz_areas[2017]
    matriz_areas['Crecimiento Porcentual (%)'] = ((matriz_areas[2025] - matriz_areas[2017]) / matriz_areas[2017].replace(0, 1) * 100).round(2)
    ranking = matriz_areas.sort_values(by='Crecimiento Absoluto', ascending=False)
    
    st.subheader("Métricas de Variación Estructural por Área (2017 vs 2025)")
    st.dataframe(
        ranking[['Crecimiento Absoluto', 'Crecimiento Porcentual (%)']].style.format({'Crecimiento Absoluto': '{:,.0f}', 'Crecimiento Porcentual (%)': '{:.2f}%'}),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fig_cabs, ax_cabs = plt.subplots(figsize=(8, 5))
        ranking['Crecimiento Absoluto'].plot(kind='barh', ax=ax_cabs, color='seagreen')
        ax_cabs.set_title("Crecimiento Absoluto de Estudiantes por Área")
        ax_cabs.invert_yaxis()
        st.pyplot(fig_cabs)
    with col2:
        fig_cpor, ax_cpor = plt.subplots(figsize=(8, 5))
        ranking['Crecimiento Porcentual (%)'].plot(kind='barh', ax=ax_cpor, color='darkorange')
        ax_cpor.set_title("Crecimiento Porcentual (%) por Área")
        ax_cpor.invert_yaxis()
        st.pyplot(fig_cpor)


        # --- MÓDULO 3: ENFOQUE SECTOR ECONÓMICO ---
elif modulo == "3. Enfoque Sector Económico":
    st.header("Enfoque Específico en las Carreras del Área Económica")
    st.markdown("Comportamiento comercial de las carreras agrupadas bajo las macroáreas de *Administración y Comercio*.")
    
    df_eco = df[df['area del conocimiento'].str.contains('ADMINISTRACION|COMERCIO', case=False, na=False)].copy()
    
    st.subheader("Evolución por Tipo de Plantel de Educación Superior")
    ev_inst_eco = df_eco.groupby(['año', 'clasificacion institucion nivel 1'])['total matricula'].sum().unstack().fillna(0)
    st.dataframe(ev_inst_eco.style.format('{:,.0f}'), use_container_width=True)
    
    col1, col2 = st.columns([6, 4])
    with col1:
        fig_eco_l, ax_eco_l = plt.subplots(figsize=(8, 5))
        ev_inst_eco.plot(kind='line', marker='o', ax=ax_eco_l, cmap='Set2')
        ax_eco_l.set_title("Matrícula en el Sector Económico según Tipo de Institución")
        st.pyplot(fig_eco_l)
        
    with col2:
        st.subheader("Cuota de Género Acumulada")
        tot_m = df_eco['total matricula mujeres'].sum()
        tot_h = df_eco['total matricula hombres'].sum()
        tot_nb = df_eco['total matricula no binarios o indefinidos'].sum()
        
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        ax_p.pie([tot_m, tot_h, tot_nb], labels=['Mujeres', 'Hombres', 'No Binario/Ind.'], autopct='%1.1f%%', colors=['plum', 'cadetblue', 'khaki'], startangle=140)
        ax_p.set_title("Distribución por Género Sector Económico")
        st.pyplot(fig_p)



        # --- MÓDULO 4: CLUSTER CONTROL DE GESTIÓN ---
elif modulo == "4. Cluster Control de Gestión":
    st.header("Carreras Afines a Ingeniería en Control de Gestión")
    st.markdown("Estudio de mercado detallado para las carreras analizadas: *Control de Gestión, Auditoría e Ingeniería Comercial*.")
    
    patron_carreras = 'CONTROL DE GESTION|AUDITORIA|AUDITOR|INGENIERIA COMERCIAL'
    df_cluster = df[df['nombre carrera'].str.contains(patron_carreras, case=False, na=False)].copy()
    
    def clasificar_cluster(nombre):
        nombre = str(nombre).upper()
        if 'CONTROL DE GESTION' in nombre: return 'Ingeniería en Control de Gestión'
        elif 'AUDITOR' in nombre or 'AUDITORIA' in nombre: return 'Auditoría / Contador Auditor'
        elif 'COMERCIAL' in nombre: return 'Ingeniería Comercial'
        return 'Otras carreras afines'
        
    df_cluster['carrera_cluster'] = df_cluster['nombre carrera'].apply(clasificar_cluster)
    
    st.subheader("Evolución Histórica de la Matrícula por Programa (2017–2025)")
    ev_cluster = df_cluster.groupby(['año', 'carrera_cluster'])['total matricula'].sum().unstack().fillna(0)
    st.dataframe(ev_cluster.style.format('{:,.0f}'), use_container_width=True)
    
    fig_clust, ax_clust = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_cluster.groupby(['año', 'carrera_cluster'])['total matricula'].sum().reset_index(), 
                 x='año', y='total matricula', hue='carrera_cluster', marker='o', linewidth=2, ax=ax_clust)
    ax_clust.set_title("Evolución de Matrícula Indexada por Clúster Profesional")
    ax_clust.set_ylabel("Número de Estudiantes")
    st.pyplot(fig_clust)
    
    st.markdown("---")
    
    col_share, col_gen_c = st.columns(2)
    with col_share:
        st.subheader("Participación de Mercado (Market Share)")
        participacion = df_cluster.groupby('clasificacion institucion nivel 3')['total matricula'].sum().reset_index()
        participacion['PARTICIPACIÓN (%)'] = ((participacion['total matricula'] / participacion['total matricula'].sum()) * 100).round(2)
        participacion = participacion.sort_values(by='total matricula', ascending=False)
        
        st.dataframe(participacion.style.format({'total matricula': '{:,.0f}', 'PARTICIPACIÓN (%)': '{:.2f}%'}), use_container_width=True)
        
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        sns.barplot(data=participacion, x='clasificacion institucion nivel 3', y='PARTICIPACIÓN (%)', hue='clasificacion institucion nivel 3', palette='viridis', legend=False, ax=ax_bar)
        plt.xticks(rotation=45, ha='right')
        ax_bar.set_title("Market Share del Clúster por Tipo de Plantel")
        st.pyplot(fig_bar)
        
    with col_gen_c:
        st.subheader("Distribución de Género en el Clúster Profesional")
        g_muj = df_cluster['total matricula mujeres'].sum()
        g_hom = df_cluster['total matricula hombres'].sum()
        g_nb = df_cluster['total matricula no binarios o indefinidos'].sum()
        
        df_gen_c = pd.DataFrame({
            'Género': ['Mujeres', 'Hombres', 'No Binario / Indefinido'],
            'Participación (%)': [round((g_muj/(g_muj+g_hom+g_nb))*100, 2), round((g_hom/(g_muj+g_hom+g_nb))*100, 2), round((g_nb/(g_muj+g_hom+g_nb))*100, 2)]
        })
        
        st.dataframe(df_gen_c.style.format({'Participación (%)': '{:.2f}%'}), use_container_width=True)
        
        fig_bar_g, ax_bar_g = plt.subplots(figsize=(8, 5.2))
        sns.barplot(data=df_gen_c, x='Género', y='Participación (%)', hue='Género', palette='magma', legend=False, ax=ax_bar_g)
        ax_bar_g.set_ylim(0, 100)
        ax_bar_g.set_title("Porcentaje de Participación por Género (Clúster de Gestión)")
        for p in ax_bar_g.patches:
            ax_bar_g.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
        st.pyplot(fig_bar_g)

# Pie de página exigido formalmente para la defensa institucional
st.markdown("---")
st.caption("**Dashboard de Defensa de Proyecto Académico** | Desarrollado con Python y Streamlit. Fuente oficial: SIES - MINEDUC.")

