from flask import Flask, jsonify, request
from sklearn.externals import joblib
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import json
import implicit

model_als = joblib.load('./model/user_assignments_model_als.pkl')
model_bayes = joblib.load('./model/user_assignments_model_bayes.pkl')
user_submissions_pivot = pd.read_csv('./model/user_submissions_pivot.csv', index_col='id_assignments')
user_submissions = pd.read_csv('./model/user_submissions.csv')

error_rec = {'Error': 'No recommendations'}
error_rel = {'Error': 'No relations'}
error_algo = {'Error': 'No algo passed'}
error_num = {'Error': 'Number out of bound'}

app = Flask(__name__)

@app.route('/')

def hello():
    return "Hello World!"

@app.route('/recommend', methods=['GET'])

def recommend():
    user_id = request.args.get('user_id')
    num = request.args.get('num')
    algo = request.args.get('algo')

    if algo == 'als':
        if user_id is None:
            return json.dumps(error_rec)

        user_columns = list(user_submissions_pivot.columns)
        user_columns = [int(i) for i in user_columns]

        if num is None:
            return json.dumps(error_num)
        else:
            try:
                user_id_index = user_columns.index(int(user_id))
                recommendations = model_als.recommend(int(user_id_index), sparse.csr_matrix(user_submissions_pivot.values), int(num))
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
    elif algo == 'bayes':
        if user_id is None:
            return json.dumps(error_rec)

        user_columns = list(user_submissions_pivot.columns)
        user_columns = [int(i) for i in user_columns]

        if num is None:
            return json.dumps(error_num)
        else:
            try:
                user_id_index = user_columns.index(int(user_id))
                recommendations = model_bayes.recommend(int(user_id_index), sparse.csr_matrix(user_submissions_pivot.values), int(num))
            except Exception:
                recommendations = None

            if recommendations is not None:
                list_of_recommended_submissions_bayes = [i[0] for i in recommendations]
                correlation = [str(i[1]) for i in recommendations]
                list_of_recommended_submissions_bayes = user_submissions['id_assignments'][list_of_recommended_submissions_bayes]
                list_of_recommended_submissions_bayes = [str(i) for i in list_of_recommended_submissions_bayes]
                list_of_recommended_submissions_bayes = dict(zip(list_of_recommended_submissions_bayes, correlation))
                # print(list_of_recommended_submissions_bayes)
                return json.dumps(list_of_recommended_submissions_bayes)
            else:
                return json.dumps(error_rec)
    else:
        return json.dumps(error_algo)
        

@app.route('/related', methods=['GET'])

def related():
    assignment_id = request.args.get('assignment_id')
    num = request.args.get('num')
    algo = request.args.get('algo')

    if algo == 'als':
        if assignment_id is None:
            return json.dumps(error_rel)
    
        if num is None:
            return json.dumps(error_num)
        else:
            try:
                assignment_id_index = list(user_submissions_pivot.index).index(int(assignment_id))
                related = model_als.similar_items(int(assignment_id_index), int(num))
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
    elif algo == 'bayes':
        if assignment_id is None:
            return json.dumps(error_rel)
    

        if num is None:
            return json.dumps(error_num)
        else:
            try:
                assignment_id_index = list(user_submissions_pivot.index).index(int(assignment_id))
                related = model_bayes.similar_items(int(assignment_id_index), int(num))
            except Exception:
                related = None

            if related is not None:
                list_of_related_submissions_bayes = [i[0] for i in related]
                correlation = [str(i[1]) for i in related]
                list_of_related_submissions_bayes = user_submissions['id_assignments'][list_of_related_submissions_bayes]
                list_of_related_submissions_bayes = [str(i) for i in list_of_related_submissions_bayes]
                list_of_related_submissions_bayes = dict(zip(list_of_related_submissions_bayes, correlation))
                # print(list_of_related_submissions_bayes)
                return json.dumps(list_of_related_submissions_bayes)
            else:
                return json.dumps(error_rel)
    else:
        return json.dumps(error_algo)

@app.route('/enquiries', methods=['POST'])

def enquiries():
    # f = open('enquiries.txt','w')
    # f.write(request)
    # f.write(request.form['name'])
    # f.write(request.form['email'])
    # f.close()
    print(request.data)
    print(request.form['name'])
    print(request.form['email'])
    print(request.form['phone'])

if __name__ == '__main__':
    app.run()

