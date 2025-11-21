import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
from tensorflow import keras


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Hiragana Classifier": hiragana_classifier_app,
        "About": about,
        "Basic example": full_app,
        "Get center coords of circles": center_circle_app,
        "Color-based image annotation": color_annotation_app,
        "Download Base64 encoded PNG": png_export,
        "Compute the length of drawn arcs": compute_arc_length,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()



def hiragana_classifier_app():
    st.header("Clasificador de Hiragana")

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
                        confidence = prediction[0][predicted_idx] * 100

                        st.success(f"**Predicción: {predicted_class}**")
                        st.metric("Confianza", f"{confidence:.2f}%")

                        # Mostrar imagen procesada
                        st.image(img_resized, caption="Imagen procesada (64x64)", width=150)

                        # Mostrar top 10 predicciones
                        st.subheader(f"Top {top_n} Predicciones")

                        # Crear DataFrame para mostrar las predicciones
                        predictions_data = []
                        for i, idx in enumerate(top_indices, 1):
                            class_name = class_labels[idx]
                            prob = prediction[0][idx] * 100
                            predictions_data.append({
                                "Rank": i,
                                "Clase": class_name,
                                "Probabilidad": f"{prob:.2f}%"
                            })

                        df_predictions = pd.DataFrame(predictions_data)
                        st.dataframe(df_predictions, hide_index=True, use_container_width=True)

                        # Gráfico de barras de probabilidades
                        st.subheader("Distribución de Probabilidades")
                        chart_data = pd.DataFrame({
                            'Clase': [class_labels[idx] for idx in top_indices],
                            'Probabilidad (%)': [prediction[0][idx] * 100 for idx in top_indices]
                        })
                        st.bar_chart(chart_data.set_index('Clase'), height=300)
            else:
                st.warning("⚠️ No hay imagen para procesar.")

        # Botón para limpiar
        


def about():
    st.markdown(
        """
    Welcome to the demo of [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas).

    On this site, you will find a full use case for this Streamlit component, and answers to some frequently asked questions.

    :pencil: [Demo source code](https://github.com/andfanilo/streamlit-drawable-canvas-demo/)
    """
    )
    st.markdown(
        """
    What you can do with Drawable Canvas:

    * Draw freely, lines, circles and boxes on the canvas, with options on stroke & fill
    * Rotate, skew, scale, move any object of the canvas on demand
    * Select a background color or image to draw on
    * Get image data and every drawn object properties back to Streamlit !
    * Choose to fetch back data in realtime or on demand with a button
    * Undo, Redo or Drop canvas
    * Save canvas data as JSON to reuse for another session
    """
    )


def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Draw on the canvas, get the drawings back to Streamlit!
    * Configure canvas in the sidebar
    * In transform mode, double-click an object to remove it
    * In polygon mode, left-click to add a point, right-click to close the polygon, double-click to remove the latest point
    """
    )

    with st.echo("below"):
        # Specify canvas parameters in application
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == "point":
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=150,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )

        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)


def center_circle_app():
    st.markdown(
        """
    Computation of center coordinates for circle drawings some understanding of Fabric.js coordinate system
    and play with some trigonometry.

    Coordinates are canvas-related to top-left of image, increasing x going down and y going right.

    ```
    center_x = left + radius * cos(angle * pi / 180)
    center_y = top + radius * sin(angle * pi / 180)
    ```
    """
    )

    with open("saved_state.json", "r") as f:
        saved_state = json.load(f)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",  # Fixed fill color with some opacity
        stroke_width=5,
        stroke_color="black",
        background_image=bg_image,
        initial_drawing=saved_state
        if st.sidebar.checkbox("Initialize with saved state", False)
        else None,
        height=600,
        width=600,
        drawing_mode="circle",
        key="center_circle_app",
    )
    with st.echo("below"):
        if canvas_result.json_data is not None:
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if len(df) == 0:
                return
            df["center_x"] = df["left"] + df["radius"] * np.cos(
                df["angle"] * np.pi / 180
            )
            df["center_y"] = df["top"] + df["radius"] * np.sin(
                df["angle"] * np.pi / 180
            )

            st.subheader("List of circle drawings")
            for _, row in df.iterrows():
                st.markdown(
                    f'Center coords: ({row["center_x"]:.2f}, {row["center_y"]:.2f}). Radius: {row["radius"]:.2f}'
                )


def color_annotation_app():
    st.markdown(
        """
    Drawable Canvas doesn't provided out-of-the-box image annotation capabilities, but we can hack something with session state,
    by mapping a drawing fill color to a label.

    Annotate pedestrians, cars and traffic lights with this one, with any color/label you want 
    (though in a real app you should rather provide your own label and fills :smile:).

    If you really want advanced image annotation capabilities, you'd better check [Streamlit Label Studio](https://discuss.streamlit.io/t/new-component-streamlit-labelstudio-allows-you-to-embed-the-label-studio-annotation-frontend-into-your-application/9524)
    """
    )
    with st.echo("below"):
        bg_image = Image.open("img/annotation.jpeg")
        label_color = (
            st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
        )  # for alpha from 00 to FF
        label = st.sidebar.text_input("Label", "Default")
        mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"

        canvas_result = st_canvas(
            fill_color=label_color,
            stroke_width=3,
            background_image=bg_image,
            height=320,
            width=512,
            drawing_mode=mode,
            key="color_annotation_app",
        )
        if canvas_result.json_data is not None:
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if len(df) == 0:
                return
            st.session_state["color_to_label"][label_color] = label
            df["label"] = df["fill"].map(st.session_state["color_to_label"])
            st.dataframe(df[["top", "left", "width", "height", "fill", "label"]])

        with st.expander("Color to label mapping"):
            st.json(st.session_state["color_to_label"])


def png_export():
    st.markdown(
        """
    Realtime update is disabled for this demo. 
    Press the 'Download' button at the bottom of canvas to update exported image.
    """
    )
    try:
        Path("tmp/").mkdir()
    except FileExistsError:
        pass

    # Regular deletion of tmp files
    # Hopefully callback makes this better
    now = time.time()
    N_HOURS_BEFORE_DELETION = 1
    for f in Path("tmp/").glob("*.png"):
        st.write(f, os.stat(f).st_mtime, now)
        if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
            Path.unlink(f)

    if st.session_state["button_id"] == "":
        st.session_state["button_id"] = re.sub(
            "\d+", "", str(uuid.uuid4()).replace("-", "")
        )

    button_id = st.session_state["button_id"]
    file_path = f"tmp/{button_id}.png"

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data.encode()).decode()
        except AttributeError:
            b64 = base64.b64encode(img_data).decode()

        dl_link = (
            custom_css
            + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
        )
        st.markdown(dl_link, unsafe_allow_html=True)


def compute_arc_length():
    st.markdown(
        """
    Using an external SVG manipulation library like [svgpathtools](https://github.com/mathandy/svgpathtools)
    You can do some interesting things on drawn paths.
    In this example we compute the length of any drawn path.
    """
    )
    with st.echo("below"):
        bg_image = Image.open("img/annotation.jpeg")

        canvas_result = st_canvas(
            stroke_color="yellow",
            stroke_width=3,
            background_image=bg_image,
            height=320,
            width=512,
            drawing_mode="freedraw",
            key="compute_arc_length",
        )
        if (
            canvas_result.json_data is not None
            and len(canvas_result.json_data["objects"]) != 0
        ):
            df = pd.json_normalize(canvas_result.json_data["objects"])
            paths = df["path"].tolist()
            for ind, path in enumerate(paths):
                path = parse_path(" ".join([str(e) for line in path for e in line]))
                st.write(f"Path {ind} has length {path.length():.3f} pixels")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    main()