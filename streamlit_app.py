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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Revisión inicial/criterios de selección","🔎 Indicadores iniciales",  "Reducción de dimensiones", "Selección de variables", "Modelos"])


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
with tab3:


    # Subtabs
    tab_pca, tab_mca = st.tabs(["PCA / Numéricas", " MCA / Categóricas"])

    # ======================================================
    # SUBTAB PCA
    # ======================================================
    with tab_pca:
        st.subheader("📊 Análisis PCA")
        
        # Umbral dinámico
        VAR_THRESHOLD = st.slider("Umbral de selección de varianza acumulada:", 0.5, 0.99, 0.80, 0.01)

        # Filtrado de variables
        vars_excluir = ["SEQN", "Diagnóstico médico de diabetes", "Diagnóstico médico de prediabetes", "Uso actual de insulina"]
        X = df.drop(columns=vars_excluir, errors="ignore")
        X_num = X.select_dtypes(include=[np.number])

        # Excluir variables con >80% NaN
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
                            annotation_text=f"{k_pca} componentes", annotation_position="top left")
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
       

        # Gráfico de barras con Plotly
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
            st.subheader(" Biplot interactivo (segmentado por Diabetes)")

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

            fig = go.Figure()
            for cat in pcs_df["Diabetes"].dropna().unique():
                subset = pcs_df[pcs_df["Diabetes"] == cat]
                fig.add_trace(go.Scatter(
                    x=subset[pc_x],
                    y=subset[pc_y],
                    mode="markers",
                    marker=dict(size=7, opacity=0.7),
                    name=str(cat),
                    hoverinfo="x+y+name"
                ))

            fig.update_layout(
                title=f"Biplot PCA ({pc_x} vs {pc_y}) - Segmentado por Diabetes",
                xaxis=dict(title=f"{pc_x} ({pca.explained_variance_ratio_[ix_x]*100:.2f}%)", zeroline=True),
                yaxis=dict(title=f"{pc_y} ({pca.explained_variance_ratio_[ix_y]*100:.2f}%)", zeroline=True),
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                legend=dict(itemsizing="constant", orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ El Biplot interactivo requiere al menos 2 componentes principales.")

    # ======================================================
    # SUBTAB MCA
    # ======================================================
    with tab_mca:

    # Filtrar categóricas
        X_cat = X.select_dtypes(exclude=[np.number]).dropna(axis=1, how="all")

        if X_cat.shape[1] == 0:
            st.warning("⚠️ No hay variables categóricas disponibles para MCA.")
        else:
            mca = MCA(n_components=5, random_state=42)
            mca_fit = mca.fit(X_cat)

            coords = mca_fit.transform(X_cat)
            coords.columns = [f"Dim{i}" for i in range(1, coords.shape[1]+1)]

            eigvals = mca_fit.eigenvalues_
            inertia = eigvals / eigvals.sum()
            inertia_cum = np.cumsum(inertia)

            st.write("### Varianza explicada (inercia)")
            df_inertia = pd.DataFrame({
                "Dimensión": coords.columns,
                "Inercia": inertia,
                "Inercia acumulada": inertia_cum
            })
            st.dataframe(df_inertia)

            # === Biplot MCA individuos ===
            st.subheader("🎯 Biplot MCA (individuos) - Segmentado por Diabetes")
            dim_x = st.selectbox("Eje X (MCA):", coords.columns, index=0)
            dim_y = st.selectbox("Eje Y (MCA):", coords.columns, index=1)

            coords_plot = coords.copy()
            coords_plot["Diabetes"] = df["Diagnóstico médico de diabetes"]

            fig = go.Figure()
            for cat in coords_plot["Diabetes"].dropna().unique():
                subset = coords_plot[coords_plot["Diabetes"] == cat]
                fig.add_trace(go.Scatter(
                    x=subset[dim_x],
                    y=subset[dim_y],
                    mode="markers",
                    marker=dict(size=7, opacity=0.7),
                    name=str(cat),
                    hoverinfo="x+y+name"
                ))

            fig.update_layout(
                title=f"Biplot MCA ({dim_x} vs {dim_y}) - Segmentado por Diabetes",
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                legend=dict(itemsizing="constant", orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)

            # === Coordenadas de categorías ===
            coords_cat = mca_fit.column_coordinates(X_cat)

            # Calcular contribución top 5 por dimensión
            top_cats = {
                dim: coords_cat[dim].abs().sort_values(ascending=False).head(5)
                for dim in coords_cat.columns
            }

            st.subheader("🔎 Categorías con mayor contribución por dimensión (MCA)")

            for dim, contrib in top_cats.items():
                fig_cat = go.Figure()
                fig_cat.add_trace(go.Bar(
                    x=contrib.values,
                    y=contrib.index,
                    orientation="h",
                    marker=dict(color="skyblue"),
                    name=dim
                ))
                fig_cat.update_layout(
                    title=f"Top 5 categorías en {dim}",
                    xaxis_title="Contribución (abs)",
                    yaxis_title="Categorías",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cat, use_container_width=True)


# =====================
# TAB 4 SELECCIÓN DE VARIABLES
# =====================


with tab4:
    st.subheader("🔎 Selección de variables y desempeño predictivo")

    # === Sliders interactivos ===
    st.write("**Ajusta los parámetros de los modelos:**")
    lasso_C = st.slider(
        "C LASSO (Regularización, menor = más regularización)",
        min_value=0.01, max_value=10.0, value=1.0, step=0.01
    )
    rf_estimators = st.slider(
        "Número de árboles Random Forest",
        min_value=100, max_value=1000, value=500, step=50
    )
    rf_max_depth = st.slider(
        "Profundidad máxima de los árboles Random Forest",
        min_value=2, max_value=20, value=6, step=1
    )

    # === Preparar datos ===
    y = df["Diagnóstico médico de diabetes"].map({"Sí": 1, "No": 0})
    X = df.drop(columns=["SEQN", "Diagnóstico médico de diabetes", "Diagnóstico médico de prediabetes", "Uso actual de insulina"])
    X_encoded = pd.get_dummies(X, drop_first=True)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.3, random_state=42
    )

    # === Modelo LASSO ===
    model_lasso = LogisticRegression(
        penalty="l1", solver="liblinear", C=lasso_C, random_state=42
    )
    model_lasso.fit(X_train, y_train)
    y_pred_lasso = model_lasso.predict(X_test)
    y_prob_lasso = model_lasso.predict_proba(X_test)[:, 1]
    coef_lasso = model_lasso.coef_[0]
    selected_features_lasso = X_encoded.columns[coef_lasso != 0]
    num_lasso_vars = len(selected_features_lasso)

    # === Modelo Random Forest ===
    model_rf = RandomForestClassifier(
        n_estimators=rf_estimators, max_depth=rf_max_depth, random_state=42, n_jobs=-1
    )
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
    importances_rf = model_rf.feature_importances_
    selected_rf_features = X_encoded.columns[importances_rf > 0]
    num_rf_vars = len(selected_rf_features)

    # === Recuadro informativo con variables seleccionadas ===
    st.info(
        f"**Resumen de selección de variables:**\n\n"
        f"- LASSO seleccionó {num_lasso_vars} variables."
        f"- Random Forest seleccionó {num_rf_vars} variables."
    )

    # === Curva ROC comparativa con Plotly ===
    fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_prob_lasso)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_lasso = auc(fpr_lasso, tpr_lasso)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_lasso, y=tpr_lasso, mode='lines', 
                                 name=f'LASSO (AUC={roc_auc_lasso:.2f})', line=dict(color='blue')))
    fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', 
                                 name=f'Random Forest (AUC={roc_auc_rf:.2f})', line=dict(color='orange')))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))
    fig_roc.update_layout(title='Curva ROC comparativa', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc, use_container_width=True)

        # === Importancia de variables con LASSO ===
    top_lasso_idx = np.argsort(np.abs(coef_lasso))[-15:]
    features_lasso = X_encoded.columns[top_lasso_idx][::-1]
    coef_vals = coef_lasso[top_lasso_idx][::-1]

    fig_lasso = go.Figure()
    fig_lasso.add_trace(go.Bar(
        x=coef_vals, y=features_lasso,
        orientation='h', name='LASSO coef',
        marker_color='green', opacity=0.7
    ))
    fig_lasso.update_layout(
        title='Importancia de variables (LASSO)',
        xaxis_title='Coeficiente',
        yaxis_title='Variables'
    )
    st.plotly_chart(fig_lasso, use_container_width=True)


    # === Importancia de variables con Random Forest ===
    top_rf_idx = np.argsort(importances_rf)[-15:]
    features_rf = X_encoded.columns[top_rf_idx][::-1]
    rf_vals = importances_rf[top_rf_idx][::-1]

    fig_rf = go.Figure()
    fig_rf.add_trace(go.Bar(
        x=rf_vals, y=features_rf,
        orientation='h', name='RF importance',
        marker_color='orange', opacity=0.7
    ))
    fig_rf.update_layout(
        title='Importancia de variables (Random Forest)',
        xaxis_title='Importancia',
        yaxis_title='Variables'
    )
    st.plotly_chart(fig_rf, use_container_width=True)

from sklearn.decomposition import PCA
from prince import MCA

def eval_pca_auc(dfX, n_components=None):
    # solo variables numéricas
    X_num = dfX.select_dtypes(include=["int64","float64","int32","float32"])
    y_ = dfX[TARGET_COL].map({"No":0,"Sí":1}).astype(int)
    if n_components is None:
        n_components = min(10, X_num.shape[1])  # limite
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipe, X_num, y_, cv=cv, scoring="roc_auc").mean(), n_components

def eval_mca_auc(dfX, n_components=None):
    # solo categóricas
    X_cat = dfX.select_dtypes(include=["object","category","bool"])
    y_ = dfX[TARGET_COL].map({"No":0,"Sí":1}).astype(int)
    if n_components is None:
        n_components = min(10, X_cat.shape[1])
    mca = MCA(n_components=n_components, random_state=RANDOM_STATE)
    X_mca = mca.fit_transform(X_cat)
    clf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(clf, X_mca, y_, cv=cv, scoring="roc_auc").mean(), n_components
