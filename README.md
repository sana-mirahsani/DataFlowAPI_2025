# DataFlowAPI_2025

Project Workflow & Collaboration

"En binôme, en faisant des branches et des PR (simuler un conflit et faire un squash de commits)"
→ Work in pairs, using Git branches and Pull Requests.
You must simulate a merge conflict and then squash commits (combine multiple commits into one before merging).

🧠 Core Development

"API HTTP en Flask / FastAPI"
→ Create an HTTP API using Flask or FastAPI.

"Un/des module(s) contenant le code métier"
→ Write one or more Python modules that contain your business logic (the core code — e.g., data processing, model training, prediction, etc.).

"Réécrire un notebook en utilisant le module. Le notebook doit être clair et présenter l’exploration que vous avez faite des données."
→ Recreate or update a Jupyter Notebook that uses your module.
The notebook should clearly show your data exploration (data analysis, visualization, etc.), not just code.

✅ Testing & Documentation

"Des tests unitaires (module + API)"
→ Write unit tests for both your module and your API.

"Une petite doc technique (comment exécuter, quel input ?, quel output?), dans un readme"
→ Create a technical documentation (in a README file):

How to run the project

What are the inputs and outputs

Any other useful setup details

💻 Interface (User Interaction)

"Une interface graphique minimaliste pour utiliser l’API (vous pouvez utiliser le swagger de FastAPI) mais pourquoi pas utiliser Streamlit"
→ Build a simple graphical interface to interact with your API.

You can use Swagger UI (built into FastAPI)

Or make a small Streamlit app as a front end

🧩 Command-Line Usage

You should have 2 or 3 command-line commands, for example:

To train the model from a CSV file
→ e.g. python train_model.py data.csv

To launch the API
→ e.g. uvicorn app:app --reload

To start the web interface (UI)
→ e.g. streamlit run app_ui.py

project_name/
│
├── data/                     # your datasets
├── notebooks/
│   └── exploration.ipynb     # your analysis notebook
│
├── src/
│   ├── data_utils.py         # data preparation functions
│   ├── model_utils.py        # model training/prediction
│   └── api.py                # Flask or FastAPI endpoints
│
├── tests/
│   ├── test_data_utils.py
│   └── test_api.py
│
├── README.md                 # how to run, inputs/outputs, etc.
├── train.py                  # command to train model
└── requirements.txt          # dependencies
