import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras


# Mapeo de romanji a hiragana
HIRAGANA_MAP = {
    'aa': 'あ', 'chi': 'ち', 'ee': 'え', 'fu': 'ふ', 'ha': 'は', 'he': 'へ',
    'hi': 'ひ', 'ho': 'ほ', 'ii': 'い', 'ka': 'か', 'ke': 'け', 'ki': 'き',
    'ko': 'こ', 'ku': 'く', 'ma': 'ま', 'me': 'め', 'mi': 'み', 'mo': 'も',
    'mu': 'む', 'na': 'な', 'ne': 'ね', 'ni': 'に', 'nn': 'ん', 'no': 'の',
    'nu': 'ぬ', 'oo': 'お', 'ra': 'ら', 're': 'れ', 'ri': 'り', 'ro': 'ろ',
    'ru': 'る', 'sa': 'さ', 'se': 'せ', 'shi': 'し', 'so': 'そ', 'su': 'す',
    'ta': 'た', 'te': 'て', 'tsu': 'つ', 'to': 'と', 'uu': 'う', 'wa': 'わ',
    'wo': 'を', 'ya': 'や', 'yo': 'よ', 'yu': 'ゆ'
}


def main():
    PAGES = {
        "Hiragana Classifier CNN": hiragana_classifier_app,
        "Hiragana Transfer Learning": hiragana_transfer_learning_app,
    }
    page = st.sidebar.selectbox("Selecciona el modelo:", options=list(PAGES.keys()))
    PAGES[page]()


def hiragana_classifier_app():
    st.header("Clasificador de Hiragana CNN desde 0")

    # Configuración del modelo
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_CHANNELS = 1

    # Clases del modelo (orden alfabético)
    class_labels = ['aa', 'chi', 'ee', 'fu', 'ha', 'he', 'hi', 'ho', 'ii',
                    'ka', 'ke', 'ki', 'ko', 'ku', 'ma', 'me', 'mi', 'mo',
                    'mu', 'na', 'ne', 'ni', 'nn', 'no', 'nu', 'oo', 'ra',
                    're', 'ri', 'ro', 'ru', 'sa', 'se', 'shi', 'so', 'su',
                    'ta', 'te', 'tsu', 'to', 'uu', 'wa', 'wo', 'ya', 'yo', 'yu']

    # Cargar modelo (usar caché para no recargarlo cada vez)
    @st.cache_resource
    def load_model():
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_cnn_hiragana_model.h5")
        try:
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            st.info(f"Ruta intentada: {model_path}")
            return None

    model = load_model()

    if model is None:
        st.warning("No se pudo cargar el modelo. Por favor verifica que el archivo existe en la carpeta models/")
        return

    st.success("Modelo cargado exitosamente")

    # Configuración del canvas
    st.sidebar.header("Configuración del Canvas")
    stroke_width = st.sidebar.slider("Grosor del trazo:", 5, 30, 15)
    canvas_size = st.sidebar.slider("Tamaño del canvas:", 200, 500, 400)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Dibuja aquí")
        # Crear canvas cuadrado para dibujar
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key="hiragana_canvas",
            display_toolbar=True,
        )

    with col2:
        st.subheader("Predicciones")
        if st.button("🗑️ Limpiar Canvas", use_container_width=True):
            st.rerun()

        # Botón para predecir
        if st.button("🔮 Predecir", type="primary", use_container_width=True):
            if canvas_result.image_data is not None:
                # Obtener imagen del canvas
                img_data = canvas_result.image_data

                # Verificar que hay contenido dibujado
                if np.sum(img_data[:, :, :3]) == img_data.shape[0] * img_data.shape[1] * 255 * 3:
                    st.warning("⚠️ El canvas está vacío. Dibuja algo primero.")
                else:
                    with st.spinner("Analizando..."):
                        # Procesar igual que en el notebook
                        img_array = np.array(img_data, dtype=np.uint8)

                        # Convertir a escala de grises (tomar canal rojo, todos son iguales)
                        img_gray = img_array[:, :, 0]

                        # Crear imagen PIL
                        img_pil = Image.fromarray(img_gray, mode='L')

                        # Redimensionar a 64x64
                        img_resized = img_pil.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)

                        # Normalizar
                        img_normalized = np.array(img_resized) / 255.0

                        # Reshape para el modelo
                        img_input = img_normalized.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

                        # Predecir
                        prediction = model.predict(img_input, verbose=0)

                        # Obtener top 10 predicciones
                        top_n = 10
                        top_indices = np.argsort(prediction[0])[-top_n:][::-1]

                        # Mostrar predicción principal
                        predicted_idx = top_indices[0]
                        predicted_class = class_labels[predicted_idx]
                        hiragana_char = HIRAGANA_MAP.get(predicted_class, '')
                        confidence = prediction[0][predicted_idx] * 100

                        st.success(f"**Predicción: {predicted_class} ({hiragana_char})**")
                        st.metric("Confianza", f"{confidence:.2f}%")

                        # Mostrar imagen procesada
                        st.image(img_resized, caption="Imagen procesada (64x64)", width=150)

                        # Mostrar top 10 predicciones
                        st.subheader(f"Top {top_n} Predicciones")

                        # Crear DataFrame para mostrar las predicciones
                        predictions_data = []
                        for i, idx in enumerate(top_indices, 1):
                            class_name = class_labels[idx]
                            hiragana = HIRAGANA_MAP.get(class_name, '')
                            prob = prediction[0][idx] * 100
                            predictions_data.append({
                                "Rank": i,
                                "Clase": f"{class_name} ({hiragana})",
                                "Probabilidad": f"{prob:.2f}%"
                            })

                        df_predictions = pd.DataFrame(predictions_data)
                        st.dataframe(df_predictions, hide_index=True, use_container_width=True)

                        # Gráfico de barras de probabilidades
                        st.subheader("Distribución de Probabilidades")
                        chart_data = pd.DataFrame({
                            'Clase': [f"{class_labels[idx]} ({HIRAGANA_MAP.get(class_labels[idx], '')})" for idx in top_indices],
                            'Probabilidad (%)': [prediction[0][idx] * 100 for idx in top_indices]
                        })
                        st.bar_chart(chart_data.set_index('Clase'), height=300)
            else:
                st.warning("⚠️ No hay imagen para procesar.")


def hiragana_transfer_learning_app():
    st.header("Clasificador de Hiragana - Transfer Learning")
    st.info("Este modelo fue entrenado usando Transfer Learning con MobileNetV2 pre-entrenada.")

    # Configuración del modelo (Transfer Learning usa RGB 128x128)
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3

    # Clases del modelo (orden alfabético)
    class_labels = ['aa', 'chi', 'ee', 'fu', 'ha', 'he', 'hi', 'ho', 'ii',
                    'ka', 'ke', 'ki', 'ko', 'ku', 'ma', 'me', 'mi', 'mo',
                    'mu', 'na', 'ne', 'ni', 'nn', 'no', 'nu', 'oo', 'ra',
                    're', 'ri', 'ro', 'ru', 'sa', 'se', 'shi', 'so', 'su',
                    'ta', 'te', 'tsu', 'to', 'uu', 'wa', 'wo', 'ya', 'yo', 'yu']

    # Cargar modelo (usar caché para no recargarlo cada vez)
    @st.cache_resource
    def load_transfer_model():
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_hiragana_transfer_model.h5")
        try:
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            st.info(f"Ruta intentada: {model_path}")
            return None

    model = load_transfer_model()

    if model is None:
        st.warning("No se pudo cargar el modelo. Por favor verifica que el archivo existe en la carpeta models/")
        return

    st.success("Modelo Transfer Learning cargado exitosamente")

    # Configuración del canvas
    st.sidebar.header("Configuración del Canvas")
    stroke_width = st.sidebar.slider("Grosor del trazo:", 5, 30, 15, key="tl_stroke")
    canvas_size = st.sidebar.slider("Tamaño del canvas:", 200, 500, 400, key="tl_canvas_size")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Dibuja aquí")
        # Crear canvas cuadrado para dibujar
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key="hiragana_transfer_canvas",
            display_toolbar=True,
        )

    with col2:
        st.subheader("Predicciones")
        if st.button("🗑️ Limpiar Canvas", use_container_width=True, key="tl_clear"):
            st.rerun()

        # Botón para predecir
        if st.button("🔮 Predecir", type="primary", use_container_width=True, key="tl_predict"):
            if canvas_result.image_data is not None:
                # Obtener imagen del canvas
                img_data = canvas_result.image_data

                # Verificar que hay contenido dibujado
                if np.sum(img_data[:, :, :3]) == img_data.shape[0] * img_data.shape[1] * 255 * 3:
                    st.warning("⚠️ El canvas está vacío. Dibuja algo primero.")
                else:
                    with st.spinner("Analizando..."):
                        # Procesar para Transfer Learning (RGB 128x128)
                        img_array = np.array(img_data, dtype=np.uint8)

                        # Convertir a escala de grises primero
                        img_gray = img_array[:, :, 0]

                        # Crear imagen PIL en escala de grises
                        img_pil = Image.fromarray(img_gray, mode='L')

                        # Convertir a RGB (3 canales) para Transfer Learning
                        img_rgb = img_pil.convert('RGB')

                        # Redimensionar a 128x128
                        img_resized = img_rgb.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)

                        # Convertir a array y normalizar
                        img_array_rgb = np.array(img_resized) / 255.0

                        # Reshape para el modelo
                        img_input = img_array_rgb.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

                        # Predecir
                        prediction = model.predict(img_input, verbose=0)

                        # Obtener top 10 predicciones
                        top_n = 10
                        top_indices = np.argsort(prediction[0])[-top_n:][::-1]

                        # Mostrar predicción principal
                        predicted_idx = top_indices[0]
                        predicted_class = class_labels[predicted_idx]
                        hiragana_char = HIRAGANA_MAP.get(predicted_class, '')
                        confidence = prediction[0][predicted_idx] * 100

                        st.success(f"**Predicción: {predicted_class} ({hiragana_char})**")
                        st.metric("Confianza", f"{confidence:.2f}%")

                        # Mostrar imagen procesada
                        st.image(img_resized, caption="Imagen procesada (128x128 RGB)", width=150)

                        # Mostrar top 10 predicciones
                        st.subheader(f"Top {top_n} Predicciones")

                        # Crear DataFrame para mostrar las predicciones
                        predictions_data = []
                        for i, idx in enumerate(top_indices, 1):
                            class_name = class_labels[idx]
                            hiragana = HIRAGANA_MAP.get(class_name, '')
                            prob = prediction[0][idx] * 100
                            predictions_data.append({
                                "Rank": i,
                                "Clase": f"{class_name} ({hiragana})",
                                "Probabilidad": f"{prob:.2f}%"
                            })

                        df_predictions = pd.DataFrame(predictions_data)
                        st.dataframe(df_predictions, hide_index=True, use_container_width=True)

                        # Gráfico de barras de probabilidades
                        st.subheader("Distribución de Probabilidades")
                        chart_data = pd.DataFrame({
                            'Clase': [f"{class_labels[idx]} ({HIRAGANA_MAP.get(class_labels[idx], '')})" for idx in top_indices],
                            'Probabilidad (%)': [prediction[0][idx] * 100 for idx in top_indices]
                        })
                        st.bar_chart(chart_data.set_index('Clase'), height=300)
            else:
                st.warning("⚠️ No hay imagen para procesar.")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Clasificador de Hiragana",
        page_icon="🇯🇵",
        layout="wide"
    )
    st.title("🇯🇵 Clasificador de Hiragana")
    st.markdown("Dibuja un carácter hiragana y el modelo lo identificará")
    main()
