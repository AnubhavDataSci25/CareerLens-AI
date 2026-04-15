import streamlit as st
import joblib
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Kaushal AI",
    page_icon="🚀",
    layout="centered"
)

# ------------------ HEADER ------------------
st.title("🚀 Kaushal AI")
st.subheader("Career Recommendation System")

st.info("⚠️ This is a prototype demo. Predictions may not be 100% accurate.", icon="ℹ️")

# ------------------ ABOUT SECTION ------------------
with st.expander("📌 About this App"):
    st.write("""
    Kaushal AI helps you:
    - 🎯 Predict suitable career paths  
    - 📊 Analyze your skills  
    - 🧠 Get AI-based recommendations  

    Fill in your details below and discover your ideal career 🚀
    """)

# ------------------ LOAD MODELS ------------------
model = joblib.load('notebook/demo_artifacts/rf_model.pkl')
ohe_encoder = joblib.load('notebook/demo_artifacts/ohe_encoder.pkl')
mlb_encoder = joblib.load('notebook/demo_artifacts/mlb_encoder.pkl')
le_encoder = joblib.load('notebook/demo_artifacts/target_career_le_encoder.pkl')

# ------------------ FORM ------------------
with st.form("career_form"):
    st.markdown("### 🧾 Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "🎓 Education",
            ["BBA", "BCA", "BSc", "BTech", "Diploma", "MBA", "MCA"]
        )

        experience = st.slider(
            "💼 Years of Experience",
            0, 12, 1
        )

        certs = st.selectbox(
            "📜 Certification",
            ["No Certification", "AWS DevOps", "AWS ML Specialty", "AWS Security",
             "AWS Solutions Architect", "Apple Swift", "Azure DevOps",
             "Azure Fundamentals", "CBAP", "CEH Ethical Hacking", "CISSP",
             "CompTIA Security+", "Coursera ML", "Coursera Marketing",
             "Coursera PM", "DeepLearning.AI", "Flutter Certified",
             "Google Analytics", "Google Associate Android",
             "Google Cloud Professional", "Google Data Analytics",
             "Google UX Design", "HubSpot Marketing", "IBM Data Analyst",
             "IBM Data Science", "Interaction Design Foundation",
             "Kubernetes CKA", "Meta Front-End Developer", "Meta Marketing",
             "Microsoft Power BI", "MongoDB Associate", "Oracle Java", "PMP",
             "Product School PM", "Scrum Master", "Tableau Desktop",
             "TensorFlow Developer", "Udemy Mobile Dev", "Udemy Web Dev",
             "freeCodeCamp"]
        )

        learning_source = st.selectbox(
            "📚 Learning Source",
            ["bootcamp", "online-course", "self-taught", "university", "workplace"]
        )

    with col2:
        interest = st.selectbox(
            "💡 Area of Interest",
            ["backend development", "business intelligence",
             "cloud infrastructure", "cybersecurity", "data engineering",
             "devops", "frontend development", "growth marketing",
             "machine learning", "mobile development", "product strategy",
             "ux research"]
        )

        projects_count = st.slider(
            "📂 Number of Projects",
            0, 10, 2
        )

        dominant_project_domain = st.selectbox(
            "🏷️ Dominant Project Domain",
            ["none", "business", "cloud", "data", "design", "devops",
             "marketing", "ml", "mobile", "security", "web"]
        )

    skills = st.multiselect(
        "🛠️ Select Your Skills",
        ["Adobe XD", "Ads", "Agile", "Android", "Ansible", "AWS", "Azure",
         "Bash", "Business Analysis", "CI/CD", "Cloud Computing",
         "Content Writing", "Copywriting", "Cryptography", "CSS",
         "Cybersecurity", "Dart", "Data Analysis", "Deep Learning",
         "Design Systems", "Digital Marketing", "Docker", "Email Marketing",
         "Ethical Hacking", "Excel", "Feature Engineering", "Figma",
         "Firebase", "Firewall", "Flutter", "Git", "Google Analytics",
         "Google Cloud", "HTML", "iOS", "Java", "JavaScript", "Jenkins",
         "Jira", "Kotlin", "Kubernetes", "Linux", "Machine Learning",
         "Microservices", "MLflow", "MongoDB", "Networking", "Next.js",
         "NodeJS", "NumPy", "OKRs", "Pandas", "Penetration Testing",
         "PostgreSQL", "Power BI", "Process Mapping", "Product Management",
         "Prometheus", "Prototyping", "Python", "PyTorch", "R", "React",
         "React Native", "Redis", "Reporting", "Requirements Gathering",
         "REST APIs", "Roadmapping", "Sass", "SEO", "SIEM", "Sketch",
         "Social Media", "Spark", "SQL", "Statistics", "Swift", "Tableau",
         "TensorFlow", "Terraform", "TypeScript", "UI/UX", "User Research",
         "Webpack", "Wireframing"]
    )

    submit = st.form_submit_button("🔍 Analyze Career")

# ------------------ PREDICTION ------------------
if submit:

    if len(skills) == 0:
        st.warning("⚠️ Please select at least one skill.")
    else:
        with st.spinner("Analyzing your profile... 🤖"):

            # Convert skills to lowercase to match encoder classes
            skills_list = [skill.lower() for skill in skills]

            input_data = {
                'education': [education],
                'experience_years': [experience],
                'projects_count': [projects_count],
                'interests': [interest],
                'certification': [certs],
                'learning_source': [learning_source],
                'dominant_project_domain': [dominant_project_domain],
                'skills': [skills_list]
            }

            input_df = pd.DataFrame(input_data)

            # Transform skills with MultiLabelBinarizer
            input_skills = mlb_encoder.transform(input_df["skills"])
            df_skills = pd.DataFrame(input_skills, columns=mlb_encoder.classes_)

            # Transform categorical columns with OneHotEncoder
            ohe_cols = ['education', 'interests', 'certification',
                        'learning_source', 'dominant_project_domain']
            input_ohe = ohe_encoder.transform(input_df[ohe_cols])
            df_ohe = pd.DataFrame(
                input_ohe,
                columns=ohe_encoder.get_feature_names_out(ohe_cols)
            )

            # Build final feature DataFrame
            final_input_df = pd.concat(
                [input_df[['experience_years', 'projects_count']],
                 df_skills, df_ohe],
                axis=1
            )

            # Prediction
            prediction = model.predict(final_input_df)
            predicted_career = le_encoder.inverse_transform(prediction)[0]

        # ------------------ OUTPUT ------------------
        st.success(f"🎯 Recommended Career: **{predicted_career}**")

        # Extra UX
        with st.expander("📊 View Input Summary"):
            summary_df = input_df.drop(columns=["skills"]).copy()
            summary_df["skills"] = ", ".join(skills)
            st.write(summary_df)

        st.markdown("---")
        st.markdown("💡 *Tip: Add more relevant skills to improve prediction accuracy*")