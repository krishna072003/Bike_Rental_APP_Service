# Import required libraries

import streamlit as st              								# Streamlit for building the web app
import pandas as pd                 								# Pandas for data handling
import plotly.express as px         								# Plotly for interactive visualizations
from sklearn.ensemble import RandomForestRegressor   						# ML model
from sklearn.model_selection import train_test_split 						# Train/test split
from sklearn.preprocessing import LabelEncoder       						# Encode categorical variables

# Configure Streamlit page

st.set_page_config(page_title="Bike Sharing Demand Prediction", layout="wide")
st.title("🚲 Bike Sharing Demand Prediction System")  						# App title

# =============================
# Load dataset
# =============================
try:
    df = pd.read_csv("Dataset.csv")   								# Load dataset from CSV file
    
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")  						# Error if file not found
    st.stop()                                    						# Stop app execution

# Replace ? with NaN (missing values)

df.replace("?", pd.NA, inplace=True)

# =============================
# Convert numeric columns safely
# =============================

numeric_cols = ["temp", "atemp", "hum", "windspeed", "cnt", "hr", "mnth", "yr", "weekday"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  					# Convert to numbers, invalid → NaN

# =============================
# Convert normalized values back to real-world scale
# =============================

def safe_scale(col, factor):
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        max_val = df[col].dropna().max()
        if pd.notna(max_val) and max_val <= 1:   						# If values are normalized (0–1)
            df[col] = df[col] * factor           						# Scale back to actual units

safe_scale("temp", 41)        									# Temperature max ~41°C
safe_scale("atemp", 50)       									# "Feeling" temperature max ~50°C
safe_scale("windspeed", 67)   									# Windspeed max ~67 km/h

# =============================
# Encode categorical columns (convert text → numbers)
# =============================

categorical_cols = ["season", "holiday", "workingday", "weathersit", "weekday"]
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  					# Encode categories
        label_encoders[col] = le                         					# Save encoder for later use

# =============================
# Feature selection
# =============================

features = [
    "season", "yr", "mnth", "hr",
    "holiday", "workingday", "weathersit",
    "weekday", "temp", "atemp", "hum", "windspeed"
]
target = "cnt"   										# Bike demand count

df = df[features + [target]].dropna()   							# Keep only relevant columns, drop missing rows

# =============================
# Safety check: stop if dataset too small
# =============================
 
if len(df) < 20:
    st.error("❌ Dataset too small after cleaning. Check Dataset.csv format.")
    st.stop()

# =============================
# Train model - Random Forest
# =============================

X = df[features]   										# Input features
y = df[target]     										# Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

st.success("✅ Model trained successfully")

# =============================
# User input UI
# =============================
st.subheader("🔧 Enter Input Conditions")
input_data = {}

# Labels for sliders

symbol_map = {
    "temp": "🌡️ Temp - (Temperature) (°C)",
    "atemp": "🤖 Atemp - (Feels Like) (°C)",
    "hum": "💧 Humidity (%)",
    "windspeed": "🌬️ Windspeed (km/h)"
}

# Month and weekday names for dropdowns

month_names = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
weekday_names = [
    "Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"
]

# Build input form dynamically

for col in features:
    if col in categorical_cols and col != "weekday":

        le = label_encoders[col]
        options = [cls for cls in le.classes_ if pd.notna(cls) and str(cls).lower() != "<na>"]
        selected = st.selectbox(col.capitalize(), options)
        input_data[col] = le.transform([selected])[0]

    elif col == "yr":
        # Radio button ensures correct year encoding
        year_selected = st.radio("Select Year", ["2011", "2012"])
        input_data[col] = 0 if year_selected == "2011" else 1

    elif col == "mnth":
        selected_month = st.selectbox("📅 Month", month_names)
        input_data[col] = month_names.index(selected_month) + 1

    elif col == "weekday":
        selected_day = st.selectbox("🗓️ Weekday", weekday_names)
        input_data[col] = weekday_names.index(selected_day)

    else:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        label = symbol_map.get(col, col.capitalize())
        input_data[col] = st.slider(label, min_val, max_val, mean_val)

# Ensure input_df matches training features exactly

input_df = pd.DataFrame([input_data], columns=features)

# =============================
# Prediction
# =============================

# Prediction button

if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Demand: **{int(prediction[0])} bikes**")

# =============================
# Interactive Graphs with Plotly
# =============================

st.subheader("📊 Interactive Visualizations")

# Average demand per month

fig1 = px.bar(df.groupby("mnth")["cnt"].mean().reset_index(),
              x="mnth", y="cnt",
              labels={"mnth": "Month", "cnt": "Average Demand"},
              title="Average Demand per Month")
fig1.update_xaxes(tickmode="array", tickvals=list(range(1,13)), ticktext=month_names)
st.plotly_chart(fig1, use_container_width=True)

# Average demand per hour

fig2 = px.line(df.groupby("hr")["cnt"].mean().reset_index(),
               x="hr", y="cnt",
               labels={"hr": "Hour", "cnt": "Average Demand"},
               title="Average Demand per Hour")
st.plotly_chart(fig2, use_container_width=True)

# Average demand per weekday

fig3 = px.bar(df.groupby("weekday")["cnt"].mean().reset_index(),
              x="weekday", y="cnt",
              labels={"weekday": "Weekday", "cnt": "Average Demand"},
              title="Average Demand per Weekday")
fig3.update_xaxes(tickmode="array", tickvals=list(range(7)), ticktext=weekday_names)
st.plotly_chart(fig3, use_container_width=True)

# Temperature vs Demand

fig4 = px.scatter(df, x="temp", y="cnt",
                  labels={"temp": "Temperature (°C)", "cnt": "Bike Demand"},
                  title="Temperature vs Bike Demand",
                  opacity=0.6)
st.plotly_chart(fig4, use_container_width=True)

st.caption("Project - Bike Sharing Demand Prediction System | Group-1: Chanakya, Krishna et al. | Random Forest Model + Streamlit + Plotly")
