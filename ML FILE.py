import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import plotly.express as px
import numpy as np

# Simulate or load city-wise waste data
data = {
    'City': ['Nagpur', 'Mumbai', 'Pune'],
    'Average_Waste_Tons_Day': [17.1, 35.2, 20.8],
    'Recycling_Rate': [0.45, 0.38, 0.41],
    'Population': [2400000, 12400000, 3120000]
}
df = pd.DataFrame(data)

# AI Linear Regression for prediction
X = df[['Average_Waste_Tons_Day', 'Recycling_Rate', 'Population']]
y = df['Average_Waste_Tons_Day'] * np.random.uniform(1.02, 1.08, X.shape[0])  # Dummy target
model = LinearRegression().fit(X, y)
df['Predicted_Tomorrow'] = model.predict(X)

# Anomaly Detection using Isolation Forest
anomaly_model = IsolationForest(contamination=0.1).fit(df[['Average_Waste_Tons_Day', 'Recycling_Rate', 'Population']])
df['Anomaly'] = anomaly_model.predict(df[['Average_Waste_Tons_Day', 'Recycling_Rate', 'Population']])

# Streamlit dashboard layout
st.title("Smart Waste Management Dashboard")
city_choice = st.selectbox("Select City", df['City'])
city_row = df[df['City'] == city_choice].iloc[0]

st.subheader(f"Waste Data for {city_choice}")
st.write(f"Average Waste: {city_row['Average_Waste_Tons_Day']} tons/day")
st.write(f"Recycling Rate: {city_row['Recycling_Rate']*100:.1f}%")
st.write(f"Population: {int(city_row['Population'])}")

st.write(f"Predicted Waste Tomorrow: {city_row['Predicted_Tomorrow']:.2f} tons")
alert = "Needs Attention" if city_row['Anomaly']==-1 else "Normal"
st.write(f"Bin Status: {alert}")

# Plotly Data Visualization
fig = px.bar(df, x='City', y='Average_Waste_Tons_Day', color='City', title="City-wise Average Waste")
st.plotly_chart(fig)

st.info("Tip: Monitor high-waste zones using IoT sensors and promote recycling awareness!")