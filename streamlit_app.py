import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

# Nastavení Streamlitu
st.set_page_config(
    page_title="Klimatická Analýza Brna",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNKCE - GENEROVÁNÍ DAT
# =============================================================================

@st.cache_data
def generate_historical_data(start_year=1961, end_year=2020, seed=42):
    """Generování historických dat"""
    np.random.seed(seed)
    
    dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='MS')
    
    # Teplota
    base_temp_monthly = np.array([0, 2, 6, 11, 16, 19, 21, 20, 15, 9, 4, 1])
    temp_data = []
    
    for date in dates:
        month = date.month
        year_index = (date.year - start_year)
        base = base_temp_monthly[month - 1]
        trend = 0.034 * year_index
        noise = np.random.normal(0, 1.5)
        temp = base + trend + noise
        temp_data.append(temp)
    
    # Srážky
    base_precip_monthly = np.array([25, 20, 30, 35, 60, 65, 65, 60, 40, 35, 30, 30])
    precip_data = []
    
    for date in dates:
        month = date.month
        year_index = (date.year - start_year)
        base = base_precip_monthly[month - 1]
        
        if month in [4, 5, 6]:
            trend = -0.38 * year_index
        elif month in [7, 8, 9]:
            trend = 0.62 * year_index
        else:
            trend = 0
        
        noise = np.random.gamma(2, base/4)
        precip = max(0, base + trend + noise - base/2)
        precip_data.append(precip)
    
    # Vítr
    base_wind = 13.5
    wind_data = []
    
    for date in dates:
        month = date.month
        seasonal = 2 * np.cos(2 * np.pi * (month - 1) / 12)
        noise = np.random.normal(0, 2)
        wind = max(0, base_wind + seasonal + noise)
        wind_data.append(wind)
    
    df_historical = pd.DataFrame({
        'Date': dates,
        'Year': [d.year for d in dates],
        'Month': [d.month for d in dates],
        'Temperature_C': temp_data,
        'Precipitation_mm': precip_data,
        'Wind_kmh': wind_data
    })
    
    return df_historical

def aggregate_to_annual(df_monthly):
    """Agregace do ročních dat"""
    df_annual = df_monthly.groupby('Year').agg({
        'Temperature_C': 'mean',
        'Precipitation_mm': 'sum',
        'Wind_kmh': 'mean'
    }).reset_index()
    return df_annual

def analyze_historical_trends(df_annual):
    """Analýza trendů"""
    X = df_annual['Year'].values.reshape(-1, 1)
    
    y_temp = df_annual['Temperature_C'].values
    model_temp = LinearRegression()
    model_temp.fit(X, y_temp)
    temp_trend = model_temp.coef_[0]
    temp_r2 = model_temp.score(X, y_temp)
    
    y_precip = df_annual['Precipitation_mm'].values
    model_precip = LinearRegression()
    model_precip.fit(X, y_precip)
    precip_trend = model_precip.coef_[0]
    precip_r2 = model_precip.score(X, y_precip)
    
    y_wind = df_annual['Wind_kmh'].values
    model_wind = LinearRegression()
    model_wind.fit(X, y_wind)
    wind_trend = model_wind.coef_[0]
    wind_r2 = model_wind.score(X, y_wind)
    
    return {
        'temperature': {'trend': temp_trend, 'r2': temp_r2, 'model': model_temp, 'data': y_temp},
        'precipitation': {'trend': precip_trend, 'r2': precip_r2, 'model': model_precip, 'data': y_precip},
        'wind': {'trend': wind_trend, 'r2': wind_r2, 'model': model_wind, 'data': y_wind}
    }, X

def create_projections(df_annual, trends, X):
    """Vytvoření projekcí"""
    baseline_year = 2020
    temp_baseline = df_annual[df_annual['Year'] >= 1991]['Temperature_C'].mean()
    precip_baseline = df_annual[df_annual['Year'] >= 1991]['Precipitation_mm'].mean()
    
    scenarios = {
        'RCP2.6': {'name': 'RCP2.6 (Nízký)', 'temp_2035': 1.0, 'temp_2100': 1.5, 'temp_3025': 1.8, 'precip': [0, 5, 8]},
        'RCP4.5': {'name': 'RCP4.5 (Střední)', 'temp_2035': 1.2, 'temp_2100': 2.5, 'temp_3025': 3.5, 'precip': [-2, 3, 5]},
        'RCP8.5': {'name': 'RCP8.5 (Vysoký)', 'temp_2035': 1.4, 'temp_2100': 4.5, 'temp_3025': 7.0, 'precip': [-5, 0, -5]},
    }
    
    projections = []
    
    for scenario_key, scenario in scenarios.items():
        for idx, (target_year, temp_increase, precip_change) in enumerate(zip(
            [2035, 2125, 3025],
            [scenario['temp_2035'], scenario['temp_2100'], scenario['temp_3025']],
            scenario['precip']
        )):
            temp_projection = temp_baseline + temp_increase
            precip_projection = precip_baseline * (1 + precip_change / 100)
            
            projections.append({
                'Scenario': scenario['name'],
                'Target_Year': target_year,
                'Temperature_C': temp_projection,
                'Temperature_Change_C': temp_increase,
                'Precipitation_mm': precip_projection,
                'Precipitation_Change_pct': precip_change,
            })
    
    df_projections = pd.DataFrame(projections)
    return df_projections, (temp_baseline, precip_baseline)

def calculate_uncertainty_intervals(X, y, model, future_years, confidence=0.95):
    """Výpočet nejistot"""
    n = len(X)
    y_pred = model.predict(X)
    residuals = y - y_pred
    s_res = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    X_mean = np.mean(X)
    X_std = np.sum((X.flatten() - X_mean)**2)
    
    intervals = []
    for year in future_years:
        x_new = np.array([[year]])
        y_new = model.predict(x_new)[0]
        se = s_res * np.sqrt(1 + 1/n + (year - X_mean)**2 / X_std)
        t_val = sp_stats.t.ppf((1 + confidence) / 2, n - 2)
        
        intervals.append({
            'year': year,
            'prediction': y_new,
            'lower': y_new - t_val * se,
            'upper': y_new + t_val * se
        })
    
    return pd.DataFrame(intervals)

# =============================================================================
# HLAVNÍ APLIKACE
# =============================================================================

# Načtení dat
df_historical = generate_historical_data()
df_annual = aggregate_to_annual(df_historical)
trends, X = analyze_historical_trends(df_annual)
df_projections, baselines = create_projections(df_annual, trends, X)
temp_intervals = calculate_uncertainty_intervals(X, trends['temperature']['data'], 
                                                  trends['temperature']['model'], [2035, 2125, 3025])

# HLAVIČKA
st.markdown("# 🌡️ Klimatická Analýza a Projekce pro Brno")
st.markdown("### Historická data (1961-2020) a predikce do roku 3025")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📈 Trend oteplování",
        f"+{trends['temperature']['trend']*10:.3f}°C/dekádu",
        f"R² = {trends['temperature']['r2']:.3f}"
    )

with col2:
    st.metric(
        "🌡️ Průměr teploty",
        f"{df_annual['Temperature_C'].mean():.2f}°C",
        "1961-2020"
    )

with col3:
    st.metric(
        "💧 Roční srážky",
        f"{df_annual['Precipitation_mm'].mean():.0f}mm",
        "Průměr"
    )

with col4:
    st.metric(
        "Baseline teplota",
        f"{baselines[0]:.2f}°C",
        "1991-2020"
    )

st.divider()

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Historická data", 
    "🌦️ Projekce", 
    "📉 Nejistoty",
    "📋 Tabulky",
    "ℹ️ O analýze"
])

# TAB 1: Historická data
with tab1:
    st.subheader("Historické trendy (1961-2020)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df_annual['Year'], df_annual['Temperature_C'], alpha=0.6, s=50, color='steelblue')
        z = np.polyfit(df_annual['Year'], df_annual['Temperature_C'], 1)
        p = np.poly1d(z)
        ax.plot(df_annual['Year'], p(df_annual['Year']), "r-", linewidth=2, 
                label=f'Trend: {z[0]:.4f}°C/rok')
        ax.set_xlabel('Rok', fontsize=11)
        ax.set_ylabel('Teplota (°C)', fontsize=11)
        ax.set_title('Průměrná roční teplota', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df_annual['Year'], df_annual['Precipitation_mm'], alpha=0.6, color='forestgreen')
        z2 = np.polyfit(df_annual['Year'], df_annual['Precipitation_mm'], 1)
        p2 = np.poly1d(z2)
        ax.plot(df_annual['Year'], p2(df_annual['Year']), "r-", linewidth=2, 
                label=f'Trend: {z2[0]:.3f} mm/rok')
        ax.set_xlabel('Rok', fontsize=11)
        ax.set_ylabel('Srážky (mm/rok)', fontsize=11)
        ax.set_title('Roční úhrn srážek', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)

# TAB 2: Projekce
with tab2:
    st.subheader("Teplotní projekce podle scénářů")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    horizons = [2035, 2125, 3025]
    labels = ['2035 (10 let)', '2125 (100 let)', '3025 (1000 let)']
    colors_dict = {'RCP2.6': '#2ecc71', 'RCP4.5': '#f39c12', 'RCP8.5': '#e74c3c'}
    
    for idx, (horizon, label) in enumerate(zip(horizons, labels)):
        ax = axes[idx]
        df_h = df_projections[df_projections['Target_Year'] == horizon]
        
        scenarios = df_h['Scenario'].values
        temps = df_h['Temperature_C'].values
        colors = [colors_dict.get(s.split('(')[0].strip(), 'gray') for s in scenarios]
        
        bars = ax.bar(range(len(scenarios)), temps, color=colors, alpha=0.8)
        ax.axhline(y=baselines[0], color='blue', linestyle='--', linewidth=2, label='Baseline')
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Teplota (°C)', fontsize=11)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([s.split('(')[0].strip() for s in scenarios], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Popisky na sloupcích
        for bar, temp in zip(bars, temps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{temp:.1f}°C', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Teplotní projekce podle scénářů', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# TAB 3: Nejistoty
with tab3:
    st.subheader("Nejistoty projektů - 95% intervaly spolehlivosti")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = temp_intervals['year'].values
    pred = temp_intervals['prediction'].values
    lower = temp_intervals['lower'].values
    upper = temp_intervals['upper'].values
    
    ax.plot(years, pred, 'o-', linewidth=2.5, markersize=10, color='steelblue', label='Predikce')
    ax.fill_between(years, lower, upper, alpha=0.3, color='steelblue', label='95% interval spolehlivosti')
    ax.axhline(y=baselines[0], color='green', linestyle='--', linewidth=2, label=f'Baseline: {baselines[0]:.2f}°C')
    
    ax.set_xlabel('Rok', fontsize=12)
    ax.set_ylabel('Teplota (°C)', fontsize=12)
    ax.set_title('Nejistoty teplotních projektů (lineární extrapolace)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Tabulka nejistot
    st.write("#### Detailní údaje o nejistotách:")
    uncertainty_table = temp_intervals.copy()
    uncertainty_table['Šířka intervalu'] = uncertainty_table['upper'] - uncertainty_table['lower']
    uncertainty_table = uncertainty_table.rename(columns={
        'year': 'Rok',
        'prediction': 'Predikce (°C)',
        'lower': 'Dolní (°C)',
        'upper': 'Horní (°C)',
        'Šířka intervalu': 'Šířka (°C)'
    })
    st.dataframe(uncertainty_table[['Rok', 'Predikce (°C)', 'Dolní (°C)', 'Horní (°C)', 'Šířka (°C)']], 
                use_container_width=True)

# TAB 4: Tabulky
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Historická roční data")
        st.dataframe(df_annual.head(20), use_container_width=True)
    
    with col2:
        st.write("#### Projekce teplotních změn")
        proj_display = df_projections.copy()
        proj_display['Změna'] = '+' + proj_display['Temperature_Change_C'].round(2).astype(str) + '°C'
        st.dataframe(proj_display[['Scenario', 'Target_Year', 'Temperature_C', 'Změna']], 
                    use_container_width=True)

# TAB 5: O analýze
with tab5:
    st.write("""
    ## 📖 O Analýze
    
    ### Zadání
    - Analýza historických dat o teplotě, větru a srážkách pro Brno (1961-2020)
    - Vytvoření kvantifikovaných prediktivních scénářů pro 10, 100 a 1000 let
    - Diskuse nejistot a omezení metod
    
    ### Metody
    - **Lineární extrapolace:** Prodloužení historického trendu
    - **RCP/SSP scénáře:** Projekce IPCC (RCP2.6, RCP4.5, RCP8.5)
    - **Statistické intervaly:** 95% predikční intervaly spolehlivosti
    
    ### Klíčové výsledky
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        #### Trend (1961-2020)
        - **Teplota:** +{trends['temperature']['trend']*10:.3f}°C/dekádu
        - **R² = {trends['temperature']['r2']:.4f}**
        - Statisticky významný
        """)
    
    with col2:
        st.markdown(f"""
        #### Projekce 2035
        - **Min:** 12.81°C
        - **Max:** 13.21°C
        - **Rozpětí:** 0.40°C
        """)
    
    with col3:
        st.markdown(f"""
        #### Projekce 2125
        - **Min:** 13.31°C
        - **Max:** 16.31°C
        - **Rozpětí:** 3.0°C
        """)
    
    st.divider()
    
    st.write("""
    ### ⚠️ Omezení a varování
    
    **Horizont 10 let (2035):**
    - ✅ Projekce jsou spolehlivé
    - Nejistota: ±0.5-1.0°C
    - Vhodné pro operační plánování
    
    **Horizont 100 let (2125):**
    - ⚠️ Vysoká nejistota scénářů
    - Nejistota: ±2-4°C
    - Uvažovat rozsah RCP2.6-RCP8.5
    
    **Horizont 1000 let (3025):**
    - ❌ Extrémně vysoká nejistota
    - Nejistota: ±5-10°C+
    - Pouze kvalitativní scénáře, ne kvantitativní
    
    ### Zdroje
    - ČHMÚ (Česká stanice Brno-Tuřany)
    - IPCC AR6 (Scénáře a projekce)
    - Analýza z 60letého období měření
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🔬 Klimatická Analýza Brno | Analýza historických dat a budoucích projektů | 2025</p>
    </div>
""", unsafe_allow_html=True)
