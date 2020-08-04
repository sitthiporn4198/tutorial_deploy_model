import pickle
from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier

# Load Prediction Model
with open('trained_model.plk','rb') as f:
    model = pickle.load(f)

app = FastAPI()
@app.get("/")
async def main():
    return 'Deploy Model Tutorial'

@app.get("/predict")
async def create_item(alcohol:float,
                        malic_acid:float,
                        ash:float,
                        alcalinity:float,
                        magnesium:float,
                        total_phenols:float,
                        flavanoids:float,
                        nonflavanoid_phenols:float,
                        proanthocyanins:float,
                        color_intensity:float,
                        hue:float,
                        OD280_OD315:float,
                        proline:float):

    # Test API Link
    # http://127.0.0.1:5000/predict/?alcohol=12.37&malic_acid=1.17&ash=1.92&alcalinity=19.6&magnesium=78&total_phenols=2.11&flavanoids=2.0&nonflavanoid_phenols=0.27&proanthocyanins=1.04&color_intensity=4.68&hue=1.12&OD280_OD315=3.48&proline=510
    # Result Class 2

    instant = [[alcohol, malic_acid, ash, alcalinity, magnesium, total_phenols, flavanoids, 
                nonflavanoid_phenols, proanthocyanins, color_intensity, hue, OD280_OD315, proline]]

    result = model.predict(instant)[0]
    return {"Wine_Class":int(result)}

if __name__ == '__main__':
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=5000, debug=True) 