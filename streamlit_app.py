import streamlit as st
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from graphviz import Digraph
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,  recall_score
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from prince import MCA


plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="deep")

# === Título general ===
st.title("NHANES DIABETES Dashboard (2021-2023)")

# === Cargar base ===
df = pd.read_csv("nhanes_limpio.csv")

# === Sidebar Filtros ===
# === Filtros en sidebar ===
st.sidebar.header("🔎 Filtros")

# Edad
min_age, max_age = int(df["Edad en años"].min()), int(df["Edad en años"].max())
age_range = st.sidebar.slider("Edad (años):", min_age, max_age, (min_age, max_age))

# Sexo
gender_options = df["Sexo"].dropna().unique().tolist()
gender_filter = st.sidebar.multiselect("Sexo:", options=gender_options, default=gender_options)

# Raza / Etnia
race_options = df["Raza/etnia"].dropna().unique().tolist()
race_filter = st.sidebar.multiselect("Raza / Etnia:", options=race_options, default=race_options)

# IMC
min_bmi, max_bmi = float(df["Índice de masa corporal"].min()), float(df["Índice de masa corporal"].max())
bmi_range = st.sidebar.slider("Índice de masa corporal (BMI):", min_bmi, max_bmi, (min_bmi, max_bmi))

# Índice de pobreza (PIR)
poverty_range = st.sidebar.slider(
    "Índice de pobreza (PIR):",
    float(df["Índice de pobreza familiar (PIR)"].min()),
    float(df["Índice de pobreza familiar (PIR)"].max()),
    (
        float(df["Índice de pobreza familiar (PIR)"].min()),
        float(df["Índice de pobreza familiar (PIR)"].max())
    )
)

# === Aplicar filtros SOLO para Tab1 y Tab2 ===
filtered_df = df[
    (df["Edad en años"].between(age_range[0], age_range[1])) &
    (df["Sexo"].isin(gender_filter)) &
    (df["Raza/etnia"].isin(race_filter)) &
    (df["Índice de masa corporal"].between(bmi_range[0], bmi_range[1])) &
    (df["Índice de pobreza familiar (PIR)"].between(poverty_range[0], poverty_range[1]))
]


# === Pestañas ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Revisión inicial/criterios de selección","🔎 Indicadores iniciales",  "Reducción de dimensiones", "Selección de variables", "Comparación PCA_MCA vs RF", "Modelos de clasificación"])


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# === Tab1 ===
with tab1:
    # === Texto introductorio en recuadro claro con letra oscura ===
    st.markdown("""
    <div style="background-color:#f9f9f9; padding:20px; border-radius:10px; color:#1a1a1a;">
        <strong>NHANES:</strong> El <strong>National Health and Nutrition Examination Survey (NHANES)</strong> es un programa de estudios de salud realizado por el <em>National Center for Health Statistics (NCHS)</em> de los <em>Centers for Disease Control and Prevention (CDC)</em> de Estados Unidos.<br><br>
        Su objetivo es evaluar el estado de salud y nutrición de la población estadounidense mediante un diseño muestral representativo a nivel nacional.<br><br>
        El estudio combina una <strong>entrevista en el hogar</strong> —en la que se recogen datos demográficos, socioeconómicos, dietarios y de salud— con un <strong>examen físico y pruebas de laboratorio</strong> realizados en un <em>Mobile Examination Center (MEC)</em>, que es una unidad clínica móvil equipada para realizar evaluaciones estandarizadas.<br><br>
        Los datos se recogen de manera continua y se publican en ciclos de <strong>dos años</strong>, lo que permite analizar tendencias en salud a lo largo del tiempo. NHANES incluye participantes de todas las edades y etnias, y sus resultados son ampliamente utilizados para la <strong>vigilancia epidemiológica</strong>, la <strong>investigación clínica</strong> y la <strong>formulación de políticas públicas en salud</strong>.

    </div>
    """, unsafe_allow_html=True)


  

   # === Diagrama de flujo ===
    st.subheader("Diagrama de Flujo del Proceso de Selección de Datos")
    dot = Digraph(comment='Flujo de Selección de Datos', format='png')

    dot.node('A', 'Base 2021-2023\nn = 11933', shape='box', style='filled', color='lightblue')
    dot.node('B', 'Incluidos\nn = 6,296', shape='box', style='filled', color='lightgreen')
    dot.node('C', 'Base final\n6072 registros', shape='box', style='filled', color='lightgreen')
    dot.node('D', 'Se excluyeron:\nNo contestaron encuesta MET (n=3073)\nMenores de 18 años (n=2523)\nEmbarazadas (n=41)', shape='box', style='filled', color='orange')
    dot.node('E', 'Se excluyeron sujetos con valores perdidos en la variable objetivo (n=224)', shape='box', style='filled', color='red')

    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('B', 'D', constraint='false')
    dot.edge('C', 'E', constraint='false')

    st.graphviz_chart(dot)

    # === Información de la base ===
    st.info(f"📌 La base de datos tiene **{df.shape[1]} variables** y **{df.shape[0]} registros**.")

    # Muestra un sample
    st.dataframe(df.head(5))

    # --- Valores faltantes ---
    faltantes = df.isnull().mean() * 100
    faltantes_df = faltantes.reset_index()
    faltantes_df.columns = ["Variable", "% Valores faltantes"]
    faltantes_df["% Valores faltantes"] = faltantes_df["% Valores faltantes"].round(2)
    faltantes_df["Nulos"] = df.isnull().sum().values
    faltantes_df = faltantes_df.sort_values(by="% Valores faltantes", ascending=False).reset_index(drop=True)

    # --- Gráfica interactiva de Plotly ---
    st.subheader("📊 Visualización interactiva de valores faltantes")
    fig_na = px.bar(
        faltantes_df.head(15),
        x="% Valores faltantes",
        y="Variable",
        orientation='h',
        text="% Valores faltantes",
        color="% Valores faltantes",
        color_continuous_scale="Viridis",
        hover_data={"Variable": True, "% Valores faltantes": True, "Nulos": True}
    )
    fig_na.update_layout(
        title="Top 15 variables con más valores faltantes",
        xaxis_title="% de valores faltantes",
        yaxis_title="Variable",
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig_na, use_container_width=True)


# ------------------------------------------------
# TAB 2: Explorador
# ------------------------------------------------
with tab2:

    # ---- Indicadores ----

    col1, col2, col3, col4 = st.columns(4)
    total = filtered_df.shape[0] if filtered_df.shape[0] > 0 else 1

    # Diabetes
    with col1:
        diabetes_count = (filtered_df["Diagnóstico médico de diabetes"] == "Sí").sum()
        diabetes_prev = (diabetes_count / total) * 100
        st.markdown(f"""
        <div style="background-color:#FFCCCC; padding:15px; border-radius:12px; text-align:center;">
            <h4>Diabetes</h4>
            <h2>{diabetes_prev:.1f}%</h2>
            <p>{diabetes_count} casos</p>
        </div>
        """, unsafe_allow_html=True)

    # Prediabetes
    with col2:
        prediabetes_count = (filtered_df["Diagnóstico médico de prediabetes"] == "Sí").sum()
        prediabetes_prev = (prediabetes_count / total) * 100
        st.markdown(f"""
        <div style="background-color:#FFE5B4; padding:15px; border-radius:12px; text-align:center;">
            <h4>Prediabetes</h4>
            <h2>{prediabetes_prev:.1f}%</h2>
            <p>{prediabetes_count} casos</p>
        </div>
        """, unsafe_allow_html=True)

    # Insulina
    with col3:
        insulina_count = (filtered_df["Uso actual de insulina"] == "Sí").sum()
        insulina_prev = (insulina_count / total) * 100
        st.markdown(f"""
        <div style="background-color:#CCE5FF; padding:15px; border-radius:12px; text-align:center;">
            <h4>Uso de Insulina</h4>
            <h2>{insulina_prev:.1f}%</h2>
            <p>{insulina_count} casos</p>
        </div>
        """, unsafe_allow_html=True)

    # Controlada (HbA1c < 7)
    with col4:
        controlada_count = (filtered_df["Hemoglobina HbA1c (%) "] < 7).sum()
        controlada_prev = (controlada_count / total) * 100
        st.markdown(f"""
        <div style="background-color:#D4EDDA; padding:15px; border-radius:12px; text-align:center;">
            <h4>Diabetes controlada</h4>
            <h2>{controlada_prev:.1f}%</h2>
            <p>{controlada_count} pacientes controlados</p>
        </div>
        """, unsafe_allow_html=True)

        

    st.subheader("📊 Análisis interactivo de variables vs Diabetes")

    # === Selección de variable ===
    variable_seleccionada = st.selectbox(
        "Selecciona una variable para visualizar su relación con Diabetes",
        options=[col for col in df.columns if col not in ["Diagnóstico médico de diabetes", "SEQN"]]
    )

    # === Detectar tipo de variable y crear gráfico ===
    
    # --- Filtrar exclusiones ---
    excluir = ["Diagnóstico médico de prediabetes", "Uso actual de insulina"]

    if variable_seleccionada not in excluir:

        if df[variable_seleccionada].dtype in ['int64', 'float64']:
            # Boxplot interactivo con Plotly
            fig = px.box(
                df,
                x="Diagnóstico médico de diabetes",
                y=variable_seleccionada,
                color="Diagnóstico médico de diabetes",
                points="all",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title=f"Distribución de {variable_seleccionada} según Diabetes"
            )
            
            # --- Prueba estadística ---
            grupos = df.groupby("Diagnóstico médico de diabetes")[variable_seleccionada].apply(list)
            if len(grupos) == 2:
                stat, p = stats.ttest_ind(grupos["Sí"], grupos["No"], nan_policy='omit')
                st.write(f"**Prueba t de Student:** p-value = {p:.4f}")
            else:
                stat, p = stats.f_oneway(*[v for v in grupos])
                st.write(f"**ANOVA:** p-value = {p:.4f}")

        else:
            # Diagrama de barras
            conteo = df.groupby([variable_seleccionada, "Diagnóstico médico de diabetes"]).size().reset_index(name="Frecuencia")
            fig = px.bar(
                conteo,
                x=variable_seleccionada,
                y="Frecuencia",
                color="Diagnóstico médico de diabetes",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title=f"Distribución de {variable_seleccionada} vs Diabetes"
            )
            
            # --- Prueba estadística ---
            tabla = pd.crosstab(df[variable_seleccionada], df["Diagnóstico médico de diabetes"])
            chi2, p, dof, expected = stats.chi2_contingency(tabla)
            st.write(f"**Chi-cuadrado:** p-value = {p:.4f}")

        # Mostrar gráfico interactivo
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"La variable **{variable_seleccionada}** está excluida del análisis.")
        


# TAB 3 - PCA / MCA
# =====================
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ======================================================
# TAB 3: PCA / MCA
# ======================================================
with tab3:
    tab_pca, tab_mca = st.tabs(["PCA / Numéricas", "MCA / Categóricas"])

    # ======================================================
    # SUBTAB PCA
    # ======================================================
    with tab_pca:
    

        # Umbral dinámico
        VAR_THRESHOLD = st.slider("Umbral de selección de varianza acumulada:", 
                                  0.5, 0.99, 0.80, 0.01)

        # Selección y limpieza de variables numéricas
        vars_excluir = ["SEQN", "Diagnóstico médico de diabetes", 
                        "Diagnóstico médico de prediabetes", "Uso actual de insulina"]
        X = df.drop(columns=vars_excluir, errors="ignore")
        X_num = X.select_dtypes(include=[np.number])

        # Excluir variables con >80% de NaN
        missing_frac = X_num.isna().mean()
        X_num = X_num.loc[:, missing_frac <= 0.8]

        # Preprocesamiento
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        Xn_tr = num_pipe.fit_transform(X_num)

        # PCA completo
        pca_full = PCA().fit(Xn_tr)
        var_ratio = pca_full.explained_variance_ratio_
        var_cum = np.cumsum(var_ratio)

        # Nº mínimo de componentes para alcanzar el umbral
        k_pca = int(np.searchsorted(var_cum, VAR_THRESHOLD) + 1)

        # PCA final
        pca = PCA(n_components=k_pca).fit(Xn_tr)
        Z_num = pca.transform(Xn_tr)

        PC_cols = [f"PC{i}" for i in range(1, k_pca+1)]
        DF_PCA = pd.DataFrame(Z_num, columns=PC_cols, index=X_num.index)

        st.write(f"[PCA] Componentes seleccionados: **{k_pca}** (umbral={VAR_THRESHOLD:.0%})")

        # Tabla de varianzas
        df_var = pd.DataFrame({
            "Componente": [f"PC{i}" for i in range(1, len(var_cum)+1)],
            "Varianza individual": np.round(var_ratio, 4),
            "Varianza acumulada": np.round(var_cum, 4)
        })
        st.dataframe(df_var)

        # =====================
        # Scree plot interactivo con Plotly
        # =====================
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(
            x=list(range(1, len(var_cum)+1)),
            y=var_cum,
            mode='lines+markers',
            name='Varianza acumulada'
        ))
        fig_scree.add_hline(y=VAR_THRESHOLD, line_dash="dash", line_color="red",
                            annotation_text=f"Umbral {VAR_THRESHOLD*100:.0f}%", annotation_position="top right")
        fig_scree.add_vline(x=k_pca, line_dash="dash", line_color="green",
                            annotation_text=f"{k_pca} comp.", annotation_position="top left")
        fig_scree.update_layout(
            title="Scree plot - PCA",
            xaxis_title="Número de componentes",
            yaxis_title="Varianza explicada acumulada",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )
        st.plotly_chart(fig_scree, use_container_width=True)

        # =====================
        # INTERACTIVO: contribución de variables con Plotly
        # =====================
        st.write("### 🔎 Explora las variables que más aportan a cada componente")

        loadings = pca.components_.T
        df_loadings = pd.DataFrame(
            loadings,
            index=X_num.columns,
            columns=[f"PC{i}" for i in range(1, k_pca+1)]
        )

        pc_choice = st.selectbox("Selecciona un componente principal:", df_loadings.columns)
        top_n = st.slider("Número de variables a mostrar:", 5, 20, 10)

        st.markdown(f"#### {pc_choice} (varianza explicada: {pca.explained_variance_ratio_[int(pc_choice[2:])-1]*100:.2f}%)")
        top_vars = df_loadings[pc_choice].abs().sort_values(ascending=False).head(top_n)

        # Gráfico de barras
        fig_top_vars = px.bar(
            top_vars.sort_values(),
            x=top_vars.sort_values().values,
            y=top_vars.sort_values().index,
            orientation='h',
            color=top_vars.sort_values().values,
            color_continuous_scale="Viridis",
            labels={"y": "Variable", "x": "Contribución (|loading|)"}
        )
        fig_top_vars.update_layout(
            title=f"Top {top_n} variables que aportan a {pc_choice}",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_top_vars, use_container_width=True)

        # =====================
        # BIPLOT INTERACTIVO PCA
        # =====================
        if k_pca >= 2:
            st.subheader("📌 Biplot interactivo (segmentado por Diabetes)")

            comp_options = [f"PC{i}" for i in range(1, k_pca+1)]
            col1, col2 = st.columns(2)
            pc_x = col1.selectbox("Eje X:", comp_options, index=0)
            pc_y = col2.selectbox("Eje Y:", comp_options, index=1)

            ix_x = int(pc_x[2:]) - 1
            ix_y = int(pc_y[2:]) - 1

            pcs_df = pd.DataFrame({
                pc_x: Z_num[:, ix_x],
                pc_y: Z_num[:, ix_y],
                "Diabetes": df["Diagnóstico médico de diabetes"]
            })

            fig = px.scatter(
                pcs_df, x=pc_x, y=pc_y, color="Diabetes",
                labels={pc_x: f"{pc_x} ({pca.explained_variance_ratio_[ix_x]*100:.2f}%)",
                        pc_y: f"{pc_y} ({pca.explained_variance_ratio_[ix_y]*100:.2f}%)"},
                opacity=0.7
            )
            fig.update_layout(
                title=f"Biplot PCA ({pc_x} vs {pc_y}) - Segmentado por Diabetes",
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                legend=dict(itemsizing="constant", orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ El Biplot interactivo requiere al menos 2 componentes principales.")

        # =====================
        # Dataset final con PCs + objetivo
        # =====================
        TARGET_COL = "Diagnóstico médico de diabetes"  # Columna objetivo
        DF_PCA_final = pd.concat(
            [df[[TARGET_COL]].reset_index(drop=True),
             DF_PCA.reset_index(drop=True)],
            axis=1
        )

        # Guardar en CSV
        DF_PCA_final.to_csv("pca_componentes.csv", index=False)
        st.success(f"✅ Guardado **pca_componentes.csv** con shape: {DF_PCA_final.shape}")

        # Mostrar vista previa
        st.dataframe(DF_PCA_final.head())

       # ======================================================
       # ======================================================
    # SUBTAB MCA
    # ======================================================
    with tab_mca:
           
        # Selección y limpieza de categóricas
        vars_excluir = ["SEQN", "Diagnóstico médico de diabetes", 
                        "Diagnóstico médico de prediabetes", "Uso actual de insulina"]
        X = df.drop(columns=vars_excluir, errors="ignore")
        X_cat = X.select_dtypes(exclude=[np.number])
        X_cat = X_cat.fillna("Missing").astype(str)

        # Matriz disyuntiva (todas las categorías)
        X_disc = pd.get_dummies(X_cat, drop_first=False)
        # Mostrar tamaño (n_filas, n_columnas)
        st.write(f"📐 Tamaño de la matriz disyuntiva: {X_disc.shape[0]} filas x {X_disc.shape[1]} columnas")

        # Umbral dinámico
        VAR_THRESHOLD = st.slider("Umbral de selección de inercia acumulada:", 
                                  0.5, 0.99, 0.80, 0.01)


        # Ajustar MCA
        import mca
        m = mca.MCA(X_disc, benzecri=True)

        eig = np.array(m.L, dtype=float).ravel()
        inertia = eig / eig.sum()
        inertia_cum = np.cumsum(inertia)

        k_mca = int(np.searchsorted(inertia_cum, VAR_THRESHOLD) + 1)

        st.write(f"[MCA] Dimensiones seleccionadas: **{k_mca}** (umbral={VAR_THRESHOLD:.0%})")

        # Tabla de inercias
        df_inertia = pd.DataFrame({
            "Dimensión": [f"DIM{i}" for i in range(1, len(inertia_cum)+1)],
            "Inercia individual": np.round(inertia, 4),
            "Inercia acumulada": np.round(inertia_cum, 4)
        })
        st.dataframe(df_inertia)

        # =====================
        # Scree plot interactivo con Plotly
        # =====================
        fig_scree_mca = go.Figure()
        fig_scree_mca.add_trace(go.Scatter(
            x=list(range(1, len(inertia_cum)+1)),
            y=inertia_cum,
            mode='lines+markers',
            name='Inercia acumulada'
        ))
        fig_scree_mca.add_hline(y=VAR_THRESHOLD, line_dash="dash", line_color="red",
                                annotation_text=f"Umbral {VAR_THRESHOLD*100:.0f}%", annotation_position="top right")
        fig_scree_mca.add_vline(x=k_mca, line_dash="dash", line_color="green",
                                annotation_text=f"{k_mca} dim.", annotation_position="top left")
        fig_scree_mca.update_layout(
            title="Scree plot - MCA",
            xaxis_title="Número de dimensiones",
            yaxis_title="Inercia acumulada",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )
        st.plotly_chart(fig_scree_mca, use_container_width=True)

        # =====================
        # Coordenadas de individuos
        # =====================
        Fs = m.fs_r(N=k_mca)
        DF_MCA = pd.DataFrame(Fs, index=X_cat.index,
                              columns=[f"DIM{i}" for i in range(1, k_mca+1)])

        # =====================
        # Contribución de variables (categorías) a las dimensiones
        # =====================
        mass = X_disc.mean(axis=0).values
        G = m.fs_c(N=k_mca)
        eig_k = eig[:k_mca]   # usar solo los primeros k_mca autovalores
        ctr = (mass[:, None] * (G**2)) / eig_k[None, :]

        ctr_pct = ctr / ctr.sum(axis=0, keepdims=True)

        DIM_cols = [f"DIM{i}" for i in range(1, k_mca+1)]
        DF_ctr_cat = pd.DataFrame(ctr_pct, index=X_disc.columns, columns=DIM_cols)

        st.write("### 📌 Contribución de categorías a las dimensiones")
        st.dataframe(DF_ctr_cat.head(20))  # mostrar primeras 20 filas

        # =====================
        # Dataset final con DIMs + objetivo
        # =====================
        TARGET_COL = "Diagnóstico médico de diabetes"
        DF_MCA_final = pd.concat(
            [df[[TARGET_COL]].reset_index(drop=True),
             DF_MCA.reset_index(drop=True)],
            axis=1
        )

        DF_MCA_final.to_csv("mca_dimensiones.csv", index=False)
        st.success(f"✅ Guardado **mca_dimensiones.csv** con shape: {DF_MCA_final.shape}")
        st.dataframe(DF_MCA_final.head())

        # =====================
        # Dataset final conjunto PCA + MCA + objetivo
        # =====================
        if "DF_PCA_final" in locals():
            DF_final = pd.concat([DF_PCA_final.reset_index(drop=True),
                                  DF_MCA.reset_index(drop=True)], axis=1)

            DF_final.to_csv("pca_mca_concat.csv", index=False)
            st.success(f"✅ Guardado **pca_mca_concat.csv** con shape: {DF_final.shape}")
            st.dataframe(DF_final.head())
        else:
            st.warning("⚠️ Aún no has corrido el bloque PCA para generar DF_PCA_final.")

# ======================================================
# TAB 4: Selección de Variables y Comparación de Métodos
# ======================================================
with tab4:
    # ----------------------------
    # Preparar datos
    # ----------------------------
    TARGET_COL = "Diagnóstico médico de diabetes"

    if TARGET_COL not in df.columns:
        st.error(f"La columna {TARGET_COL} no está en el dataset")
    else:
        df_model = df.dropna(subset=[TARGET_COL])
        y = LabelEncoder().fit_transform(df_model[TARGET_COL])

        vars_excluir = ["SEQN", "Diagnóstico médico de diabetes", 
                        "Diagnóstico médico de prediabetes", 
                        "Uso actual de insulina"]
        X = df_model.drop(columns=vars_excluir, errors="ignore")
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.median(numeric_only=True))

        total_vars = X.shape[1]  # número total de variables disponibles

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # ----------------------------
        # RandomForest
        # ----------------------------
        rf = RandomForestClassifier(n_estimators=500, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict_proba(X_test)[:, 1]

        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        auc_rf = auc(fpr_rf, tpr_rf)

        import_rf = pd.DataFrame({
            "Variable": X.columns,
            "Importancia": rf.feature_importances_,
            "Método": "RandomForest"
        }).sort_values("Importancia", ascending=False)

        # ----------------------------
        # LASSO
        # ----------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lasso = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000, random_state=42
        )
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict_proba(X_test_scaled)[:, 1]

        fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_pred_lasso)
        auc_lasso = auc(fpr_lasso, tpr_lasso)

        import_lasso = pd.DataFrame({
            "Variable": X.columns,
            "Importancia": np.abs(lasso.coef_[0]),
            "Método": "LASSO"
        }).sort_values("Importancia", ascending=False)

        # ----------------------------
        # Curva ROC
        # ----------------------------
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines",
                                     name=f"RandomForest (AUC={auc_rf:.3f})"))
        fig_roc.add_trace(go.Scatter(x=fpr_lasso, y=tpr_lasso, mode="lines",
                                     name=f"LASSO (AUC={auc_lasso:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="gray"), showlegend=False))
        fig_roc.update_layout(title="Curva ROC: RandomForest vs LASSO",
                              xaxis_title="1 - Especificidad (FPR)",
                              yaxis_title="Sensibilidad (TPR)",
                              template="plotly_white")
        st.plotly_chart(fig_roc, use_container_width=True)

        # ----------------------------
        # Sliders para umbral de selección
        # ----------------------------
        st.subheader("⚙️ Ajuste de Umbrales de Selección")
        umbral_rf = st.slider("Percentil de importancia mínima (RandomForest)", 
                              0.0, 1.0, 0.75, 0.05)
        umbral_lasso = st.slider("Coeficiente mínimo absoluto (LASSO)", 
                                 0.0, 0.5, 0.01, 0.01)

        # Selección según umbral
        # (calcular percentil ANTES de filtrar)
        valor_umbral_rf = import_rf["Importancia"].quantile(umbral_rf)
        selected_rf = import_rf[import_rf["Importancia"] >= valor_umbral_rf]["Variable"].tolist()
        selected_lasso = import_lasso[import_lasso["Importancia"] > umbral_lasso]["Variable"].tolist()

        n_rf = len(selected_rf)
        n_lasso = len(selected_lasso)

        st.subheader("📌 Resumen de Variables")
        col1, col2, col3 = st.columns(3)
        col1.metric("Variables Totales", total_vars)
        col2.metric("Variables seleccionadas RF", n_rf)
        col3.metric("Variables seleccionadas LASSO", n_lasso)

        ganador = "RandomForest" if auc_rf >= auc_lasso else "LASSO"
        st.success(f"🏆 El método ganador según AUC es: **{ganador}**")

        # ----------------------------
        # Slider para número de variables a mostrar
        # ----------------------------
        top_n = st.slider("Número de variables a mostrar en los gráficos", 
                          min_value=5, max_value=30, value=15, step=1)

        # ----------------------------
        # Gráfico individual RF
        # ----------------------------
        fig_rf = px.bar(
            import_rf.head(top_n),
            x="Importancia", y="Variable", orientation="h",
            title=f"RandomForest - Importancia de Variables (Top {top_n})", color="Importancia"
        )
        fig_rf.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_rf, use_container_width=True)

        # ----------------------------
        # Gráfico individual LASSO
        # ----------------------------
        fig_lasso = px.bar(
            import_lasso.head(top_n),
            x="Importancia", y="Variable", orientation="h",
            title=f"LASSO - Importancia de Variables (Top {top_n})", color="Importancia"
        )
        fig_lasso.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_lasso, use_container_width=True)

        # ----------------------------
        # Dataset reducido con el método ganador
        # ----------------------------
        st.subheader("💾 Dataset reducido con variables seleccionadas")

        if ganador == "RandomForest":
            selected_vars = selected_rf
        else:
            selected_vars = selected_lasso

        df_reducido = pd.concat([df_model[[TARGET_COL]].reset_index(drop=True),
                                 X[selected_vars].reset_index(drop=True)], axis=1)

        st.write(f"El dataset reducido contiene **{df_reducido.shape[1]-1} variables** + la variable objetivo")
        st.dataframe(df_reducido.head(10))  # muestra primeras 20 filas

# ======================================================
# TAB 5: Comparación PCA vs RandomForest
# ======================================================
with tab5:
    st.subheader("📊 Comparación PCA vs RandomForest")

    if 'DF_PCA_final' not in locals():
        st.warning("⚠️ Primero ejecuta el bloque PCA (TAB 3) para generar DF_PCA_final.")
    else:
        # Preparar datos PCA
        X_pca = DF_PCA_final.drop(columns=[TARGET_COL])
        y_pca = LabelEncoder().fit_transform(DF_PCA_final[TARGET_COL])
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
            X_pca, y_pca, test_size=0.3, stratify=y_pca, random_state=42
        )

        rf_pca = RandomForestClassifier(n_estimators=500, random_state=42)
        rf_pca.fit(X_train_pca, y_train_pca)
        y_pred_pca = rf_pca.predict_proba(X_test_pca)[:, 1]

        fpr_pca, tpr_pca, _ = roc_curve(y_test_pca, y_pred_pca)
        auc_pca = auc(fpr_pca, tpr_pca)

        # RandomForest original (usando todas las variables)
        X_rf = X
        y_rf = y
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X_rf, y_rf, test_size=0.3, stratify=y_rf, random_state=42
        )
        rf_full = RandomForestClassifier(n_estimators=500, random_state=42)
        rf_full.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf_full.predict_proba(X_test_rf)[:, 1]

        fpr_rf, tpr_rf, _ = roc_curve(y_test_rf, y_pred_rf)
        auc_rf = auc(fpr_rf, tpr_rf)

        # ----------------------------
        # Curva ROC comparativa
        # ----------------------------
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines",
                                     name=f"RandomForest (Todas las vars, AUC={auc_rf:.3f})"))
        fig_roc.add_trace(go.Scatter(x=fpr_pca, y=tpr_pca, mode="lines",
                                     name=f"PCA (Componentes, AUC={auc_pca:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="gray"), showlegend=False))
        fig_roc.update_layout(title="ROC: PCA vs RandomForest",
                              xaxis_title="1 - Especificidad (FPR)",
                              yaxis_title="Sensibilidad (TPR)",
                              template="plotly_white")
        st.plotly_chart(fig_roc, use_container_width=True)

        # ----------------------------
        # Resumen y ganador
        # ----------------------------
        ganador_final = "RandomForest" if auc_rf >= auc_pca else "PCA"
        st.write(f"**AUC PCA:** {auc_pca:.3f}")
        st.write(f"**AUC RandomForest:** {auc_rf:.3f}")
        st.success(f"🏆 Método ganador según AUC: **{ganador_final}**")
# ======================================================
# TAB 6: Clasificadores con RandomizedSearchCV
# ======================================================
with tab6:
    st.subheader("🔹 Métodos de clasificación usando Dataset Reducido (RandomForest)")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # ----------------------------
    # Dataset reducido
    # ----------------------------
    df_clf = df_reducido.copy()
    X = df_clf.drop(columns=[TARGET_COL])
    y = LabelEncoder().fit_transform(df_clf[TARGET_COL])

    # ----------------------------
    # Convertir booleanas a int
    # ----------------------------
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    # ----------------------------
    # Columnas num y cat
    # ----------------------------
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # ----------------------------
    # Preprocesamiento
    # ----------------------------
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # ----------------------------
    # Balanceo de clases
    # ----------------------------
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    st.write("### Conteo de muestras por clase después del balanceo")
    class_counts = pd.Series(y_res).value_counts()
    fig_count = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        labels={'x': 'Clase', 'y': 'Cantidad de muestras'},
        title='Conteo de muestras por clase (balanceado)'
    )
    st.plotly_chart(fig_count, use_container_width=True)

    # ----------------------------
    # Split train/test
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
    )

    # ----------------------------
    # Modelos y grids
    # ----------------------------
    models = {
        "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=-1), {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }),
        "ExtraTrees": (ExtraTreesClassifier(random_state=42, n_jobs=-1), {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }),
        "HistGB": (HistGradientBoostingClassifier(random_state=42), {
            "classifier__max_iter": [100, 200, 500],
            "classifier__max_depth": [None, 5, 10],
            "classifier__learning_rate": [0.01, 0.1, 0.2]
        }),
        "LogReg": (LogisticRegression(penalty='l2', solver='liblinear', max_iter=2000, random_state=42), {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"]
        }),
        "SVM_Linear": (SVC(kernel='linear', probability=True, random_state=42), {
            "classifier__C": [0.01, 0.1, 1, 10]
        })
    }

    results = {}

    # ----------------------------
    # Entrenamiento
    # ----------------------------
    for name, (model, param_grid) in models.items():
        st.write(f"## 🔹 {name}")

        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        rs = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=15,
            scoring='f1_macro',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        rs.fit(X_train, y_train)
        best_model = rs.best_estimator_

        # Predicciones
        y_pred = best_model.predict(X_test)

        # Reporte
        st.text("📌 Reporte de clasificación")
        st.text(classification_report(y_test, y_pred))

        # Matriz de confusión
        st.text("📌 Matriz de confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
        ax.set_title(f"Matriz de confusión - {name}")
        st.pyplot(fig_cm)

        st.success(f"✅ Mejor hiperparámetro encontrado: {rs.best_params_}")

        # ROC AUC
        y_score = None
        clf = best_model.named_steps['classifier']
        try:
            if hasattr(clf, "predict_proba"):
                y_score = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_score = best_model.decision_function(X_test)
        except Exception:
            y_score = None

        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_val = roc_auc_score(y_test, y_score)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_val).plot()
            plt.title(f"Curva ROC (binaria) - {name}")
            st.pyplot(plt.gcf())

        results[name] = {
            "best_estimator": best_model,
            "best_params": rs.best_params_,
            "roc_auc": auc_val if y_score is not None else np.nan
        }


