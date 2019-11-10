import pickle
import numpy as np

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from model import OilExtractionModel

app = Flask(__name__)
api = Api(app)

model_path = "models/OilExtraction.pkl"
with open(model_path, "rb") as model_in:
    model = pickle.load(model_in)

dataset_path = "data/dummy_dataset.pkl"
with open(dataset_path, "rb") as data_in:
    dataset = pickle.load(data_in)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument("tower_id")


class OilExtractionPrediction(Resource):
    def get(self):
        args = parser.parse_args()
        tower_id = int(args["tower_id"])

        tower_features = dataset[dataset.uid == tower_id].drop("uid", axis=1)
        tower_prediction = int(model.predict(tower_features)[0])
        
        return {"prediction": tower_prediction}
        
api.add_resource(OilExtractionPrediction, "/")

if __name__ == '__main__':
    app.run(debug=True)
