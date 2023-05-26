import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text

df1 = pd.read_csv('dice_com-job_us_sample.csv')
st.set_page_config(page_title="Get Jobs", page_icon="https://prepinsta.com/wp-content/uploads/2022/07/placement-preparation-websites.webp")

st.sidebar.title("Project Members")
st.sidebar.write("- Snehal Vikhe BEITB264")
st.sidebar.write("- Rutuja More")
st.sidebar.write("- Sejal Pawar")
st.sidebar.write("- Akanksha Unhale")

st.title("👔Get Company!")
st.subheader("Here you'll be recommended with the best companies")
st.markdown('<br>', unsafe_allow_html=True)  # Add line break
# st.write(df1["jobtitle"].value_counts())

# <------------------ LOGIC -------------------->
df1 = df1.dropna()
df1.reset_index(drop=True, inplace=True)  # Reset the DataFrame index

feature = df1["skills"].tolist()
tfidf = text.TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(df1.index, index=df1['jobtitle']).drop_duplicates()


def jobs_recommendation(Title, similarity=similarity):
    index = indices[Title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[::], reverse=True)
    similarity_scores = similarity_scores[0:5]
    newsindices = [i[0] for i in similarity_scores]
    return df1[['company', 'jobdescription']].iloc[newsindices]


# <------------------ LOGIC -------------------->

values = ('Lead DevOps Engineer', "Java Developer")
option = st.selectbox(
    'Select the Role in the Dropdown Menu:',
    ('Lead DevOps Engineer', "UI Architect", "Business Analyst","Java Developer","Project Manager","Systems Engineer","Full Stack Developer", "PHP Developer", "Python Developer", "QA Engineer"))


result = jobs_recommendation(option)
# st.write(len(result))
for i in range(len(result)):
    st.markdown('<br>', unsafe_allow_html=True)  # Add line break
    st.subheader(f"Job {i+1}")
    company_name = result['company'].iloc[i]
    st.markdown(f"**Company name:** <span style='color:#ff5733'>{company_name}</span>", unsafe_allow_html=True)
    with st.expander("See Job Description"):
        st.write(result['jobdescription'].iloc[i])
    st.markdown('<br>', unsafe_allow_html=True)  # Add line break
