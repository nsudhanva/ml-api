from flask import Flask, jsonify, request
from sklearn.externals import joblib
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import json
import implicit

implicit_model = joblib.load('./model/user_assignments_model.pkl')
user_submissions_pivot = pd.read_csv('./model/user_submissions_pivot.csv', index_col='id_assignments')
user_submissions = pd.read_csv('./model/user_submissions.csv')

error_rec = {'Error': 'No recommendations'}
error_rel = {'Error': 'No relations'}
error_num = {'Error': 'Number out of bound'}

app = Flask(__name__)

@app.route("/")

def hello():
    return "Hello World!"

@app.route('/recommend', methods=['GET'])

def recommend():
    user_id = request.args.get('user_id')
    num = request.args.get('num')

    if user_id is None:
        return json.dumps(error_rec)

    user_columns = list(user_submissions_pivot.columns)
    user_columns = [int(i) for i in user_columns]
    user_id_index = user_columns.index(int(user_id))

    if num is None:
        return json.dumps(error_num)
    else:
        try:
            recommendations = implicit_model.recommend(int(user_id_index), sparse.csr_matrix(user_submissions_pivot.values), int(num))
        except Exception:
            recommendations = None

        if recommendations is not None:
            list_of_recommended_submissions_als = [i[0] for i in recommendations]
            correlation = [str(i[1]) for i in recommendations]
            list_of_recommended_submissions_als = user_submissions['id_assignments'][list_of_recommended_submissions_als]
            list_of_recommended_submissions_als = [str(i) for i in list_of_recommended_submissions_als]
            list_of_recommended_submissions_als = dict(zip(list_of_recommended_submissions_als, correlation))
            # print(list_of_recommended_submissions_als)
            return json.dumps(list_of_recommended_submissions_als)
        else:
            return json.dumps(error_rec)
        

@app.route('/related', methods=['GET'])

def related():
    assignment_id = request.args.get('assignment_id')
    num = request.args.get('num')

    if assignment_id is None:
        return json.dumps(error_rel)
   
    assignment_id_index = list(user_submissions_pivot.index).index(int(assignment_id))

    if num is None:
        return json.dumps(error_num)
    else:
        try:
            related = implicit_model.similar_items(int(assignment_id_index), int(num))
        except Exception:
            related = None

        if related is not None:
            list_of_related_submissions_als = [i[0] for i in related]
            correlation = [str(i[1]) for i in related]
            list_of_related_submissions_als = user_submissions['id_assignments'][list_of_related_submissions_als]
            list_of_related_submissions_als = [str(i) for i in list_of_related_submissions_als]
            list_of_related_submissions_als = dict(zip(list_of_related_submissions_als, correlation))
            # print(list_of_related_submissions_als)
            return json.dumps(list_of_related_submissions_als)
        else:
            return json.dumps(error_rel)

if __name__ == '__main__':
    app.run()

