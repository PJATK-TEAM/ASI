import streamlit as st
from streamlit_option_menu import option_menu
import requests

st.set_page_config(page_title="AmIHealthy")

# Horizontal menu
selected2 = option_menu(None, ["Home", "Upload", "History", 'Settings'],
    icons=['house', 'cloud-upload', "list-task", 'gear'],
    menu_icon="cast", default_index=0, orientation="horizontal")

# Home
if selected2 == "Home":
    try:
        with st.spinner("Connecting to backend..."):
            response = requests.get("http://backend:8000/")

        if response.status_code == 200:
            st.markdown("## 🔍 Pneumonia Detector")

            st.markdown("---")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("""
                ### Welcome to the Pneumonia Detector!

                This application helps you determine whether an image contains signs of pneumonia.

                **How to use:**
                1. Navigate to the **Upload** tab
                2. Upload an image file (JPG, JPEG, or PNG)
                3. Wait for the AI to classify your image
                4. View the detailed results
                """)

            with col2:
                st.markdown("### Classification Types:")
                st.success("🫁 Pneumonia")
                st.warning("🩺 Normal")

        else:
            st.markdown("### 🔴 System Status: Ups... Something went wrong :/")
    except requests.exceptions.ConnectionError:
        st.markdown("### 🔴 System Status: Ups... Something went wrong :/")

#Upload
elif selected2 == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Classifying..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post("http://backend:8000/classify", files=files)

                if response.status_code == 200:
                    result = response.json()

                    if "prediction" in result:
                        pneumonia_prob = result["prediction"]["Pneumonia"]
                        normal_prob = result["prediction"]["Normal"]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🫁 Pneumonia")
                            st.progress(pneumonia_prob)
                            st.markdown(f"**{pneumonia_prob * 100:.2f}%**")

                        with col2:
                            st.markdown("### 🩺 Normal")
                            st.progress(normal_prob)
                            st.markdown(f"**{normal_prob * 100:.2f}%**")

                        st.markdown("---")
                        if pneumonia_prob > normal_prob:
                            st.success(f"This is most likely **Pneumonia** 🫁 ({pneumonia_prob * 100:.2f}%)")
                        else:
                            st.warning(f"This is most likely a **Normal** 🩺 ({normal_prob * 100:.2f}%)")
                    else:
                        st.write(result)
                else:
                    st.error(f"Failed to classify the image: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection error: Backend server is not running. Please start the backend server first.")

# History
elif selected2 == "History":
    response = requests.get("http://backend:8000/history")
    if response.status_code == 200:
        history = response.json()
        st.markdown("## 🕓 Prediction History")

        if not history:
            st.info("No classification history found.")
        else:
            for item in history:
                file_name = item["file_name"]
                timestamp = item["timestamp"]
                result = item["result"]

                pneumonia = result.get("Pneumonia", 0)
                normal = result.get("Normal", 0)

                st.markdown(f"**🗂 File:** {file_name}")
                st.markdown(f"🕒 {timestamp}")
                st.progress(pneumonia)
                st.markdown(f"🫁 **Pneumonia**: {pneumonia * 100:.2f}%")
                st.progress(normal)
                st.markdown(f"🩺 **Normal**: {normal * 100:.2f}%")
                st.markdown("---")

# Settings
elif selected2 == "Settings":
    response = requests.get("http://backend:8000/settings")
    if response.status_code == 200:
        settings = response.json()
        for item in settings:
            st.write(f"File: {item['file_name']}, Result: {item['result']}")
    else:
        # st.write("Failed to fetch settings")
        st.write("Sorry, the feature hasn't been implemented yet.")
