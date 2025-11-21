# Proyecto-DeepLearning

# Predicción utilizando el modelo MNIST aplicado hacia el alfabeto japónes


![alt text](image.png)

Esta aplicación de Streamlit permite dibujar caracteres hiragana en un canvas y obtener predicciones en tiempo real usando modelos de Deep Learning.

## Ejecución
```
   py -m streamlit run UI/app.py
```

## Modelos Disponibles

La aplicación incluye dos modelos diferentes:

1. **Hiragana Classifier CNN**: Modelo CNN entrenado desde cero (`best_cnn_hiragana_model.h5`)
2. **Hiragana Transfer Learning**: Modelo entrenado usando Transfer Learning (`best_hiragana_transfer_model.h5`)

## Características

- **Canvas de dibujo interactivo**: Dibuja caracteres hiragana con el ratón
- **Predicciones en tiempo real**: Los modelos identifican el carácter dibujado
- **Top 10 predicciones**: Muestra las 10 predicciones más probables con sus porcentajes
- **Visualización de probabilidades**: Gráfico de barras mostrando la distribución
- **Configuración ajustable**: Cambia el grosor del trazo y tamaño del canvas
- **Comparación de modelos**: Prueba el mismo dibujo con ambos modelos

## Requisitos

```bash
pip install streamlit streamlit-drawable-canvas tensorflow pillow numpy pandas
```

## Cómo ejecutar

Desde la carpeta `UI/`, ejecuta:

```bash
streamlit run app.py
```

Luego, en la barra lateral, selecciona la opción que desees de la lista de páginas.

## Uso

### Modelo CNN desde cero

1. En la barra lateral, selecciona **"Hiragana Classifier CNN"**
2. Dibuja un carácter hiragana en el canvas blanco usando el ratón
3. Haz clic en el botón **"🔮 Predecir"** para obtener las predicciones
4. Revisa las predicciones del modelo:
   - Predicción principal con su confianza
   - Tabla con las Top 10 predicciones
   - Gráfico de barras de probabilidades
5. Haz clic en **"🗑️ Limpiar Canvas"** para borrar y dibujar de nuevo

### Modelo Transfer Learning

1. En la barra lateral, selecciona **"Hiragana Transfer Learning"**
2. Sigue los mismos pasos que con el modelo CNN
3. Compara los resultados con los del modelo CNN

## Configuración

En la barra lateral puedes ajustar:
- **Grosor del trazo**: Entre 5 y 30 píxeles (por defecto 15)
- **Tamaño del canvas**: Entre 200 y 500 píxeles (por defecto 400)

## Caracteres soportados

Ambos modelos pueden reconocer los siguientes 46 caracteres hiragana:

あ (aa), ち (chi), え (ee), ふ (fu), は (ha), へ (he), ひ (hi), ほ (ho), い (ii),
か (ka), け (ke), き (ki), こ (ko), く (ku), ま (ma), め (me), み (mi), も (mo),
む (mu), な (na), ね (ne), に (ni), ん (nn), の (no), ぬ (nu), お (oo), ら (ra),
れ (re), り (ri), ろ (ro), る (ru), さ (sa), せ (se), し (shi), そ (so), す (su),
た (ta), て (te), つ (tsu), と (to), う (uu), わ (wa), を (wo), や (ya), よ (yo), ゆ (yu)

## Notas técnicas

- Ambos modelos esperan imágenes de 64x64 píxeles en escala de grises
- El canvas automáticamente redimensiona y procesa tu dibujo
- El procesamiento es idéntico al usado en el notebook demo (fondo blanco, trazo negro)
- Los modelos se cargan con caché para mejor rendimiento

# Integrantes
- Ricardo Chuy, 221007
- Eunice Mata, 21231
- Andre Jo, 22199