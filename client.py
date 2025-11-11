import requests

# URL of the Flask API endpoint
url = 'http://127.0.0.1:8000/predict'

# Path to the image file you want to send
file_path = './test/aa/drawing_20250805_081550.jpg'  # Replace this with the path to your image

# Open the image file and send it in the POST request
with open(file_path, 'rb') as file:
    files = {'file': (file.name, file, 'image/jpeg')}  # 'image/jpeg' can be replaced with the actual MIME type
    response = requests.post(url, files=files)

# Print the response from the Flask API
print(response.json())
