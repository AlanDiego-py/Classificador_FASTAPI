# Classificação de Flores Íris com FastAPI

Este projeto é uma aplicação web para classificação de flores do conjunto de dados Íris utilizando aprendizado de máquina (RandomForest) e FastAPI. O usuário informa as medidas da flor e recebe a espécie prevista.

## Funcionalidades

- Interface web simples para entrada dos dados das flores.
- Predição da espécie da flor (Setosa, Versicolor ou Virginica).
- Visualização do resultado diretamente na página.
- Testes automatizados com pytest.

## Estrutura de Pastas

```
classificação/
├── app/
│   ├── static/
│   │   └── style.css
│   └── templates/
│       ├── index.html
│       └── result.html
├── src/
│   └── iris.py
├── tests/
│   └── test_iris.py
├── main.py
├── iris_model.pkl
├── requirements.txt
└── README.md
```

## Como executar

1. **Instale as dependências:**
   ```
   pip install -r requirements.txt
   ```

2. **Treine o modelo (se necessário):**
   Execute o script `src/iris.py` para gerar o arquivo `iris_model.pkl` caso ele não exista.

3. **Inicie o servidor FastAPI:**
   ```
   uvicorn main:app --reload
   ```
   Acesse [http://localhost:8000](http://localhost:8000) no navegador.

## Testes

Para rodar os testes automatizados:

```
pytest
```

## Requisitos

- Python 3.8+
- FastAPI
- scikit-learn
- joblib
- numpy
- pytest

## Personalização

- O modelo pode ser ajustado em `src/iris.py`.
- O estilo visual pode ser alterado em `app/static/style.css`.
- Os templates HTML estão em `app/templates/`.

## Licença

Este projeto é livre para uso acadêmico e pessoal.

---