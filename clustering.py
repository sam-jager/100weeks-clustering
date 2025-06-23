import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="100WEEKS Clustering", layout="wide")

st.title("Clustering Analysis 100WEEKS")

# Automatische refresh elke 10 minuten
st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sam-jager/100weeks-clustering/main/central-tableau-export-2.0.csv"
    return pd.read_csv(url, low_memory=False)

@st.cache_resource
def get_encoder():
    return OneHotEncoder(sparse_output=False, handle_unknown='ignore')

@st.cache_resource
def get_scaler():
    return StandardScaler()

countries = ["GHA", "RWA", "UGA", "CIV", "KEN"]
country = st.selectbox("Select a country", countries)

if country:
    df = load_data()

    if "Country" not in df.columns:
        st.error("De kolom 'Country' ontbreekt in de dataset.")
        st.stop()

    df = df[df['Country'].isin(countries)]
    df_country = df[df["Country"] == country]

    if df_country.empty:
        st.warning(f"Geen data beschikbaar voor {country}.")
        st.stop()

    columns = [
        "Groupnr", "Round", "Country", "childmortality", "childmortalitytime",
        *[f"foodsecurity{i}" for i in range(1, 10)],
        *[f"foodsecurity{i}freq" for i in range(1, 10)],
        "fuelcooking", "sourcelighting", "watersource", "timewatersource_1",
        "timewatersourceunit", "Toiletfacility", "materialroof", "materialfloor",
        "materialwallsext", "assetsmatrix2_7", "assetsmatrix2_14", "assetsmatrix2_16",
        "assetsmatrix1_23", "assetsmatrix3_14", "assetsmatrix3_16", "assetsmatrix2_12",
        "assetsmatrix3_22", *[f"HHMschool_{n}" for n in range(1, 6)],
        *[f"HHMschoolnow_{n}" for n in range(1, 6)],
        *[f"HHMschoolcompl_{n}" for n in range(1, 6)],
        "school", "schoolcompleted", "savinghowmuch_1", "savinghowmuch_2",
        "savinghowmuch_3", "savingstotal_1", "debt", "debtamount_1", "debtnote",
        *[f"anxiety{i}" for i in range(1, 8)],
        "psychwellbeing_1", "psychwellbeing_3", "psychwellbeing_5", "psychwellbeing2_5",
        "jealousy", "jealousywhat", *[f"livestocknumbers_{i}" for i in [1,13,3,4,5,6,11,8,9,7,2,10]],
        "assetsmatrix1_4", "assetsmatrix1_5", "assetsmatrix1_22", "assetsmatrix2_15",
        "assetsmatrix2_8", "assetsmatrix3_17", "assetsmatrix2_17", "assetsmatrix2_18",
        "assetsmatrix2_19", "assetsmatrix2_11", "assetsmatrix3_15", "assetsmatrix3_23",
        "occupationmain", "ownsland_scto", "meetings1", "moneywithdraw", "moneyproblems"
    ]

    df_country = df_country[[col for col in columns if col in df_country.columns]]

    df_country = df_country[~df_country['Round'].isin(['Onboarding', '6', '6.0']) & df_country['Round'].notna()]
    df_country['Round'] = pd.to_numeric(df_country['Round'], errors='coerce')
    df_country = df_country[df_country['Round'] % 1 == 0]
    df_country['Round'] = df_country['Round'].astype(int).astype(str)
    df_country = df_country[df_country['Round'].isin(['0', '1', '2', '3', '100', '102'])]
    df_country['Round'] = df_country['Round'].astype(float).astype(int).astype(str)
    df_country = df_country.sort_values(by='Round', ascending=True)

    numerical = ["savinghowmuch_1", "savinghowmuch_2", "savinghowmuch_3", "savingstotal_1", "debtamount_1", "timewatersource_1"]
    ordered_categorical = [*[f"foodsecurity{i}freq" for i in range(1, 10)], *[f"anxiety{i}" for i in range(1, 8)], "psychwellbeing_1", "psychwellbeing_3", "psychwellbeing_5", "psychwellbeing2_5"]
    categorical = ["fuelcooking", "sourcelighting", "watersource", "Toiletfacility", "materialroof", "materialfloor", "materialwallsext", *[f"HHMschoolcompl_{n}" for n in range(1, 6)], "schoolcompleted", "livestocknumbers_1", *[f"livestocknumbers_{i}" for i in [1, 13, 3, 4, 5, 6, 11, 8, 9, 7, 2, 10]], "occupationmain"]
    binary = ["childmortality", *[f"foodsecurity{i}" for i in range(1, 10)], *[f"HHMschool_{n}" for n in range(1, 6)], *[f"HHMschoolnow_{n}" for n in range(1, 6)], "school", "debt", "jealousy", "assetsmatrix1_4", "assetsmatrix1_5", "assetsmatrix1_22", "assetsmatrix2_15", "assetsmatrix2_8", "assetsmatrix3_17", "assetsmatrix2_17", "assetsmatrix2_18", "assetsmatrix2_19", "assetsmatrix2_11", "assetsmatrix3_15", "assetsmatrix3_23", "meetings1", "moneywithdraw", "moneyproblems"]
    binary_neg = ["debt", "foodsecurity1", "foodsecurity2", "foodsecurity3", "foodsecurity4", "foodsecurity5", "foodsecurity6", "foodsecurity7", "foodsecurity8", "foodsecurity9", "childmortality", "jealousy", "assetsmatrix1_4", "assetsmatrix1_5", "assetsmatrix1_22", "assetsmatrix2_15", "assetsmatrix2_8", "assetsmatrix3_17", "assetsmatrix2_17", "assetsmatrix2_18", "assetsmatrix2_19", "assetsmatrix2_11", "assetsmatrix3_15", "assetsmatrix3_23"]
    binary_pos = ["HHMschoolnow_1", "HHMschoolnow_2", "HHMschoolnow_3", "HHMschoolnow_4", "HHMschoolnow_5", "school", "meetings1", "moneywithdraw", "moneyproblems"]

    @st.cache_data
    def preprocess(df, round_value):
        df_r = df[df['Round'] == round_value].copy()
        round_binary = [col for col in binary if col in df_r.columns]
        round_categorical = [col for col in categorical if col in df_r.columns]
        round_ordered = [col for col in ordered_categorical if col in df_r.columns]
        round_numerical = [col for col in numerical if col in df_r.columns]

        df_r = df_r[round_binary + round_categorical + round_ordered + round_numerical + ['Groupnr']].copy()

        for col in round_binary:
            df_r[col] = pd.to_numeric(df_r[col], errors='coerce')

        for col in binary_neg:
            if col in df_r.columns:
                df_r[col] = df_r[col].replace({2.0: 1.0, 1.0: 0.0})

        for col in binary_pos:
            if col in df_r.columns:
                df_r[col] = df_r[col].replace({1.0: 1.0, 2.0: 0.0})

        for col in round_categorical + round_ordered:
            if col in df_r.columns:
                df_r[col] = df_r[col].astype(str).where(df_r[col].notna(), np.nan)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = ohe.fit_transform(df_r[round_categorical])
        cat_feature_names = ohe.get_feature_names_out()
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df_r.index)
        dummy_to_original = {new: new.split('_')[0] for new in cat_feature_names}

        df_features = df_r[round_binary + round_ordered + round_numerical]
        df_all = pd.concat([df_features, cat_df], axis=1)
        df_all = df_all.apply(pd.to_numeric, errors='coerce')
        df_grouped = df_all.groupby(df_r['Groupnr']).mean().dropna(axis=1, how='any').dropna(axis=0, how='any')

        return df_grouped, dummy_to_original

    def cluster_and_plot(df_grouped, dummy_to_original, round_nr):
        scaler = get_scaler()
        X_scaled = scaler.fit_transform(df_grouped)
    
        best_k = 2
        best_score = -1
        for k in range(2, min(5, len(X_scaled))):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k = k
                best_score = score
    
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
    
        group_ids = df_grouped.index.astype(str)
        group_df = pd.DataFrame({
            'x': X_pca[:, 0],
            'y': X_pca[:, 1],
            'Cluster': labels.astype(str),
            'Groupnr': group_ids
        })
    
        # Zoekbalk
        selected_group = st.text_input(f"Zoek een Groupnr voor ronde {round_nr} (optioneel):")
        highlight_group = group_df[group_df['Groupnr'] == selected_group] if selected_group in group_ids.values else pd.DataFrame()
    
        # Plot
        fig = px.scatter(
            group_df,
            x='x',
            y='y',
            color='Cluster',
            hover_name='Groupnr',
            labels={'x': 'Socio-economic status', 'y': 'Living conditions and facilities'},
            title=f"Clustering for round {round_nr}"
        )
    
        # Accentueer geselecteerde groep
        if not highlight_group.empty:
            fig.add_scatter(
                x=highlight_group['x'],
                y=highlight_group['y'],
                mode='markers+text',
                marker=dict(size=15, color='black', symbol='x'),
                text=highlight_group['Groupnr'],
                name='Geselecteerde groep'
            )
    
        st.plotly_chart(fig, use_container_width=True)


    available_rounds = ['0', '2', '100']
    for r in available_rounds:
        st.subheader(f"Round {r}")
        try:
            df_grouped, dummy_to_original = preprocess(df_country, r)
            if not df_grouped.empty:
                cluster_and_plot(df_grouped, dummy_to_original, r)
            else:
                st.info(f"Geen bruikbare data voor ronde {r}.")
        except Exception as e:
            st.error(f"Fout bij verwerken van ronde {r}: {e}")

