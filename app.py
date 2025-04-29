#!/bin/python3
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Set page config
st.set_page_config(page_title="Job Finder & Salary Predictor", layout="centered")

# Load model
try:
    model = joblib.load('linear_model.pkl')
except FileNotFoundError:
    st.error("Model file 'linear_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Load dataset
try:
    df = pd.read_csv('final_data.csv')
except FileNotFoundError:
    st.error("Dataset 'final_data.csv' not found. Please ensure it's in the same directory.")
    st.stop()

# Preprocessing
df['Reviews'] = df['Reviews'].astype(str)
df['Reviews'] = df['Reviews'].str.replace(r'\D', '', regex=True).replace('', '0').astype(int)

for col in ['Title', 'Location', 'Experience', 'Skills', 'Company', 'Job_Post_History']:
    df[col] = df[col].astype(str)

def extract_avg_experience(exp):
    if pd.isna(exp) or str(exp).strip() in ['', 'nan']:
        return 0
    exp = str(exp).replace('Yrs', '').strip()
    if '-' in exp:
        try:
            low, high = map(int, exp.split('-'))
            return (low + high) / 2
        except:
            return 0
    try:
        return float(exp)
    except:
        return 0

df['Experience_Avg'] = df['Experience'].apply(extract_avg_experience)
df['Ratings'] = df['Ratings'].fillna(df['Ratings'].median())

for col in ['Location', 'Title', 'Skills', 'Company', 'Job_Post_History']:
    df[col] = df[col].replace(['', 'nan'], 'Unknown')

# Label Encoding
le_location = LabelEncoder()
le_company = LabelEncoder()
le_skills = LabelEncoder()
le_title = LabelEncoder()
le_job_post = LabelEncoder()

df['Location_Encoded'] = le_location.fit_transform(df['Location'])
df['Company_Encoded'] = le_company.fit_transform(df['Company'])
df['Skills_Encoded'] = le_skills.fit_transform(df['Skills'])
df['Title_Encoded'] = le_title.fit_transform(df['Title'])
df['Job_Post_Encoded'] = le_job_post.fit_transform(df['Job_Post_History'])

# Feature Scaling
scaler = StandardScaler()
training_features = df[['Experience_Avg', 'Reviews', 'Ratings', 'Location_Encoded',
                        'Company_Encoded', 'Skills_Encoded', 'Title_Encoded', 'Job_Post_Encoded',
                        'Experience_Avg', 'Reviews', 'Ratings', 'Experience_Avg', 'Reviews', 'Ratings']]
scaler.fit(training_features)

# App UI
st.title("🔎 Job Finder and 💰 Salary Prediction App")
st.markdown("Enter job details to find matching jobs and predict salaries in INR:")

job_title_input = st.selectbox("🧑‍💼 Job Title", options=['Any'] + sorted(df['Title'].unique()))
location_input = st.selectbox("🏙 Location", options=['Any'] + sorted(df['Location'].unique()))
years_experience = st.slider("📅 Years of Experience", min_value=0, max_value=20, value=5)
skills_input = st.selectbox("🛠 Skills", options=['Any'] + sorted(df['Skills'].unique()))
company_input = st.selectbox("🏢 Company", options=['Any'] + sorted(df['Company'].unique()))
search_button = st.button("🚀 Search Matching Jobs")

# Prepare input
def prepare_input(title, location, experience, skills, company, df, le_location, le_company, le_skills, le_title, le_job_post, scaler=None):
    filtered_df = df.copy()
    if title != 'Any':
        filtered_df = filtered_df[filtered_df['Title'] == title]
    if location != 'Any':
        filtered_df = filtered_df[filtered_df['Location'] == location]
    if company != 'Any':
        filtered_df = filtered_df[filtered_df['Company'] == company]
    if skills != 'Any':
        filtered_df = filtered_df[filtered_df['Skills'] == skills]

    default_df = filtered_df if not filtered_df.empty else df
    avg_reviews = default_df['Reviews'].median()
    avg_ratings = default_df['Ratings'].median()
    avg_experience = float(experience)

    location_encoded = le_location.transform([location])[0] if location in le_location.classes_ else default_df['Location_Encoded'].mode()[0]
    company_encoded = le_company.transform([company])[0] if company in le_company.classes_ else default_df['Company_Encoded'].mode()[0]
    skills_encoded = le_skills.transform([skills])[0] if skills in le_skills.classes_ else default_df['Skills_Encoded'].mode()[0]
    title_encoded = le_title.transform([title])[0] if title in le_title.classes_ else default_df['Title_Encoded'].mode()[0]
    job_post_encoded = default_df['Job_Post_Encoded'].mode()[0]

    input_data = np.array([[
        avg_experience, avg_reviews, avg_ratings, location_encoded,
        company_encoded, skills_encoded, title_encoded, job_post_encoded,
        avg_experience, avg_reviews, avg_ratings, avg_experience, avg_reviews, avg_ratings
    ]])

    if scaler:
        input_data = scaler.transform(input_data)
    return input_data

# Find matching jobs (no top_n limit)
def find_matching_jobs(df, title, location, skills, company, experience):
    filtered_df = df.copy()
    if title != 'Any':
        filtered_df = filtered_df[filtered_df['Title'].str.contains(title, case=False, na=False)]
    if location != 'Any':
        filtered_df = filtered_df[filtered_df['Location'].str.contains(location, case=False, na=False)]
    if skills != 'Any':
        filtered_df = filtered_df[filtered_df['Skills'].str.contains(skills, case=False, na=False)]
    if company != 'Any':
        filtered_df = filtered_df[filtered_df['Company'].str.contains(company, case=False, na=False)]

    if filtered_df.empty:
        return filtered_df

    filtered_df['Exp_Diff'] = abs(filtered_df['Experience_Avg'] - experience)
    filtered_df['Score'] = filtered_df['Exp_Diff']

    return filtered_df.sort_values('Score')

# Process search
if search_button:
    if not job_title_input and location_input == 'Any' and not skills_input and company_input == 'Any' and years_experience == 0:
        st.warning("⚠️ Please provide at least one search criterion.")
    else:
        input_data = prepare_input(job_title_input, location_input, years_experience, skills_input, company_input,
                                   df, le_location, le_company, le_skills, le_title, le_job_post)

        try:
            predicted_salary_usd = model.predict(input_data)[0]
            conversion_rate = 83.0
            predicted_salary_inr = predicted_salary_usd * conversion_rate
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()

        matching_jobs = find_matching_jobs(df, job_title_input, location_input, skills_input, company_input, years_experience)

        if matching_jobs.empty:
            st.warning("⚠️ No matching jobs found. Try different keywords or broaden your search.")
        else:
            st.success(f"✅ Found {len(matching_jobs)} matching jobs!")
            st.subheader("Top Matching Jobs")

            for idx, row in matching_jobs.iterrows():
                with st.container():
                    st.markdown(f"**🏢 {row['Company']} | 📍 {row['Location']}**")
                    st.markdown(f"**🧑‍💼 Title:** {row['Title']}")
                    st.markdown(f"**📅 Experience Required:** {row['Experience']}")

                    st.markdown(f"**⭐ Ratings:** {row['Ratings']} stars | 📝 Reviews: {row['Reviews']}")
                    st.markdown(f"**🛠 Skills:** {row['Skills']} | 💰 **Predicted Salary:** ₹{predicted_salary_inr:,.2f}")

                    if 'URL' in row and pd.notna(row['URL']):
                        job_url = row['URL']
                        st.markdown(f'<a href="{job_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:8px 16px;border:none;border-radius:4px;">View Job</button></a>', unsafe_allow_html=True)

                    st.markdown("---")  # separator between jobs
