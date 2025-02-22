import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

def model_prediction(test_image):
    model = keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Dictionary of disease solutions
disease_solutions = {
    'Apple___Apple_scab': "Solution: To control apple scab, apply fungicides such as sulfur or captan early in the season. Rake and remove fallen leaves to reduce fungal spores in the environment. Ensure proper tree spacing for air circulation and consider planting resistant varieties.",
    'Apple___Black_rot': "Solution: Prune and destroy infected branches, fruit, and leaves. Apply copper-based fungicides during early bloom and petal fall. Keep the orchard clean by removing mummified fruit and improving air circulation.",
    'Apple___Cedar_apple_rust': "Solution: Remove nearby juniper trees that serve as an alternate host for the fungus. Prune affected branches, and apply fungicides like myclobutanil or mancozeb at bud break to prevent infection.",
    'Apple___healthy': "Solution: No disease detected. Maintain proper tree care, including regular watering, fertilization, and pruning to promote healthy growth and disease resistance.",
    'Blueberry___healthy': "Solution: No disease detected. Keep soil pH between 4.5 and 5.5, apply mulch to retain moisture, and prune for better air circulation.",
    'Cherry___Powdery_mildew': "Solution: Apply fungicides such as sulfur, neem oil, or potassium bicarbonate to control the spread. Ensure proper pruning to improve airflow and reduce humidity around the plant.",
    'Cherry___healthy': "Solution: No disease detected. Regularly prune trees, provide adequate watering, and fertilize properly to keep them healthy.",
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot': "Solution: Remove and destroy infected leaves, practice crop rotation, and apply fungicides like strobilurins to prevent further spread.",
    'Corn___Common_rust_': "Solution: Plant rust-resistant corn varieties and apply fungicides like propiconazole if needed. Maintain proper plant spacing to improve air circulation.",
    'Corn___Northern_Leaf_Blight': "Solution: Use resistant hybrids, apply fungicides at early infection stages, and practice crop rotation to reduce pathogen buildup in the soil.",
    'Corn___healthy': "Solution: No disease detected. Ensure proper irrigation, fertilization, and weed control to maintain plant health.",
    'Grape___Black_rot': "Solution: Remove and destroy infected leaves, fruit, and stems. Apply fungicides such as mancozeb or myclobutanil at bloom and repeat as necessary. Ensure proper vineyard management, including good airflow and sanitation.",
    'Grape___Esca_(Black_Measles)': "Solution: Prune out infected wood, improve soil drainage, and avoid overwatering. Apply protective fungicides and maintain good vineyard hygiene.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Solution: Remove and dispose of affected leaves, avoid overhead irrigation, and apply fungicides like copper-based sprays to manage the disease.",
    'Grape___healthy': "Solution: No disease detected. Regularly prune vines, provide balanced fertilization, and monitor for pests and diseases.",
    'Orange___Haunglongbing_(Citrus_greening)': "Solution: Remove and destroy infected trees to prevent spread. Control the Asian citrus psyllid with insecticides and encourage biological control. Use certified disease-free planting material.",
    'Peach___Bacterial_spot': "Solution: Remove and destroy infected branches and leaves. Apply copper-based fungicides during dormancy and early season. Avoid overhead watering to minimize moisture on leaves.",
    'Peach___healthy': "Solution: No disease detected. Maintain good orchard hygiene, prune for airflow, and ensure balanced fertilization.",
    'Pepper,_bell___Bacterial_spot': "Solution: Remove affected leaves and destroy them. Apply copper-based fungicides and use resistant varieties. Avoid working with plants when they are wet to prevent the spread.",
    'Pepper,_bell___healthy': "Solution: No disease detected. Ensure proper spacing, adequate watering, and regular fertilization for optimal plant health.",
    'Potato___Early_blight': "Solution: Remove and destroy affected leaves. Apply fungicides like chlorothalonil or mancozeb. Rotate crops and avoid overhead irrigation to prevent moisture buildup.",
    'Potato___Late_blight': "Solution: Apply fungicides containing copper or chlorothalonil. Remove and destroy infected plants immediately. Avoid overcrowding and ensure proper drainage.",
    'Potato___healthy': "Solution: No disease detected. Maintain good soil health, rotate crops, and provide adequate nutrients to support strong growth.",
    'Raspberry___healthy': "Solution: No disease detected. Prune regularly, ensure good airflow, and provide balanced fertilization.",
    'Soybean___healthy': "Solution: No disease detected. Use crop rotation, manage soil fertility, and monitor for early signs of disease.",
    'Squash___Powdery_mildew': "Solution: Apply sulfur-based fungicides, prune infected leaves, and ensure proper spacing for ventilation. Water plants at the base to reduce leaf moisture.",
    'Strawberry___Leaf_scorch': "Solution: Remove infected leaves, apply fungicides if necessary, and ensure proper watering and fertilization practices.",
    'Strawberry___healthy': "Solution: No disease detected. Maintain proper watering and fertilization for optimal plant growth.",
    'Tomato___Bacterial_spot': "Solution: Remove and destroy infected leaves. Apply copper-based fungicides and avoid overhead watering.",
    'Tomato___Early_blight': "Solution: Prune lower infected leaves, apply fungicides, and mulch around plants to prevent soil splash.",
    'Tomato___Late_blight': "Solution: Apply fungicides immediately and remove infected plants to prevent the disease from spreading rapidly.",
    'Tomato___Leaf_Mold': "Solution: Improve air circulation, remove affected leaves, and apply fungicides like copper-based sprays.",
    'Tomato___Septoria_leaf_spot': "Solution: Prune affected areas, apply fungicides, and ensure proper spacing between plants for airflow.",
    'Tomato___Spider_mites_Two-spotted_spider_mite': "Solution: Spray with miticides, introduce predatory insects like ladybugs, and use insecticidal soaps.",
    'Tomato___Target_Spot': "Solution: Apply fungicides and remove infected leaves promptly to prevent spread.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Solution: Remove infected plants, control whiteflies with insecticides, and use resistant tomato varieties.",
    'Tomato___Tomato_mosaic_virus': "Solution: Remove affected plants and sanitize gardening tools to prevent transmission.",
    'Tomato___healthy': "Solution: No disease detected. Ensure good soil health, proper irrigation, and regular monitoring for pests and diseases."
}


# Sidebar Layout
st.sidebar.title("🌱 Plant Disease Detection System")
st.sidebar.markdown("""
    ## 🌿 Explore Our Plant Disease Detection System
    Detect plant diseases effortlessly using our powerful AI-driven system. Upload an image, and we'll analyze it for you in a heartbeat! 

    ### 📚 Navigate through the pages:
    - **Home:** Welcome message and easy-to-follow instructions to get you started.
    - **About:** Learn about the dataset and how our system works.
    - **Disease Recognition:** Upload an image of a plant and let us predict the disease for you.

    🌟 **Let's make plant care smarter and easier!** 🌟
""")
app_mode = st.sidebar.selectbox("Choose a page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌱 Welcome to the Plant Disease Detection System")
    
    # Add a sleek home page image
    image_path = "planthomepgimage.jpg"  # Ensure this file exists
    try:
        image = Image.open(image_path)
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.error("Oops! It looks like we couldn't find the image. Please check the file path.")
    
    st.markdown("""
    Welcome to the **Plant Disease Recognition System!** 🌿🌍

    ### How It Works:
    1. **Upload an Image:** Go to the **Disease Recognition** page and upload an image of a plant leaf that might have a disease.
    2. **Automatic Analysis:** Our system will quickly process the image and analyze it for any diseases.
    3. **Get Results:** You’ll receive a clear result with detailed predictions, so you can act fast.

    ### Why Choose Us?
    - **Accurate Predictions:** Powered by cutting-edge machine learning technology.
    - **User-Friendly Interface:** Intuitive for everyone, no technical experience needed.
    - **Instant Results:** Get your predictions in seconds, saving valuable time.

    ### 🌟 Ready to Start?
    Head to the **Disease Recognition** page and upload your image to kick off the diagnosis process.
    """)
    
# About Project
elif app_mode == "About":
    st.header("💡 About the Project")
    st.markdown("""
    #### 📊 Dataset Overview
    Our dataset contains **87K RGB images** of both healthy and diseased crop leaves, split into **38 categories**. These images are carefully classified to help our system deliver accurate disease predictions.

    #### Key Stats:
    - **Training Set:** 70,295 images 📈
    - **Validation Set:** 17,572 images 📊
    - **Test Set:** 33 images 📉

    #### 🌾 Disease Categories:
    Our system can detect a wide variety of plant diseases including, but not limited to:
    - **Apple Scab 🍏**
    - **Tomato Leaf Mold 🍅**
    - **Potato Early Blight 🥔**
    - And many more!

    #### 📑 Dataset Details:
    - The images are high-quality and come from real-world farming conditions, providing a diverse set of scenarios for detection.
    - The system is trained to recognize diseases accurately, helping farmers and plant enthusiasts diagnose problems fast and effectively.
    """)



# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # File Upload
    test_image = st.file_uploader("Choose an Image of a Plant Leaf", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", width=400, use_container_width=True)
    
    # Predict Button
    if st.button("Predict Disease"):
        if test_image:
            # st.balloons()  # Show snow animation for effect
            result_index = model_prediction(test_image)

            # Plant disease classes
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Gray_leaf_spot',
                            'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
                            'Grape___Esca', 'Grape___Leaf_blight', 'Grape___healthy', 'Orange___Citrus_greening',
                            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper___Bacterial_spot', 'Pepper___healthy',
                            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                            'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot',
                            'Tomato___Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            
            class_name = class_names[result_index]
            st.success(f"Prediction: This is most likely a **{class_name}**.")

            # Display solution based on the predicted disease
            solution = disease_solutions.get(class_name, "No specific solution available.")
            st.markdown(f"{solution}")

        else:
            st.warning("Please upload an image first!")
