from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


from fastapi.staticfiles import StaticFiles
from pathlib import Path

templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Carregar o modelo treinado
model = joblib.load('iris_model.pkl')

# Inicializar o FastAPI
app = FastAPI()

# Configurar o Jinja2 para renderizar templates HTML
templates = Jinja2Templates(directory="app/templates")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Definir o esquema de dados para a entrada
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
class IrisOutput(BaseModel):
    species: str

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# endpoint para prever a espécie da íris
@app.post("/predict", response_model=IrisOutput)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    # Preparar os dados para a predição
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Fazer a predição
    prediction = model.predict(input_data)
    
    # Mapear o índice da predição para o nome da espécie
    iris = load_iris()
    species = iris.target_names[prediction][0]
    
    return templates.TemplateResponse(
        "result.html", 
        {   
            "request": request,
            "prediction": species,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width, 
         
         },
        )

# Rodar o aplicativo com: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)



