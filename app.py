from flask import Flask, jsonify, request
from sklearn.externals import joblib
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import json
import implicit

implicit_model = joblib.load('./model/user_assignments_model.pkl')
user_assignments_sparse = sparse.load_npz('./model/user_assignments.npz')
user_submissions = pd.read_csv('./model/user_submissions.csv')

app = Flask(__name__)

@app.route("/")

def hello():
    return "Hello World!"

@app.route('/recommend', methods=['GET'])

def recommend():
    user_id = request.args.get('user_id')
    recommendations = implicit_model.recommend(int(user_id), user_assignments_sparse, N=20)
    list_of_recommended_submissions = [i[0] for i in recommendations]
    list_of_recommended_submissions = user_submissions['id_assignments'][list_of_recommended_submissions].to_dict()
    return json.dumps(list_of_recommended_submissions)

@app.route('/related', methods=['GET'])

def related():
    assignment_id = request.args.get('assignment_id')
    related = implicit_model.similar_items(int(assignment_id), N=20)
    
    list_of_related_submissions = [i[0] for i in related]
    list_of_related_submissions = user_submissions['user_id_submissions'][list_of_related_submissions].to_dict()
    return json.dumps(list_of_related_submissions)

if __name__ == '__main__':
    app.run()

