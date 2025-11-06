import requests

GEMINI_API_KEY = 'AIzaSyDjcmpdzo4HiOwi7Ct-NF5m2PUqe0uWTFc'

# Listar modelos disponibles
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
response = requests.get(url)

print("Status Code:", response.status_code)
print("\nRespuesta completa:")
print(response.text)

if response.status_code == 200:
    data = response.json()
    print("\n\nModelos disponibles que soportan generateContent:")
    if 'models' in data:
        for model in data['models']:
            if 'generateContent' in model.get('supportedGenerationMethods', []):
                print(f"  - {model['name']}")