from flask import Flask, jsonify, request
from sklearn.externals import joblib
import scipy.sparse as sparse
import json

app = Flask(__name__)

if __name__ == '__main__':
    app.run()

@app.route("/")

def hello():
    return "Hello World!"

@app.route('/recommend', methods=['GET'])

def recommend():
    implicit_model = joblib.load('./model/user_submissions_model.pkl')
    user_assignments_sparse = sparse.load_npz('./model/user_assignments.npz')
    user_id = request.args.get('user_id')
    recommendations = implicit_model.recommend(int(user_id), user_assignments_sparse, N=20)
    recommendations = dict(recommendations)
    recommendations = {str(key): str(value) for key, value in recommendations.items()}
    return json.dumps(recommendations)

@app.route('/related', methods=['GET'])

def related():
    implicit_model = joblib.load('./model/user_submissions_model.pkl')
    user_assignments_sparse = sparse.load_npz('./model/user_assignments.npz')
    submission_id = request.args.get('submission_id')
    related = implicit_model.similar_items(int(submission_id), N=20)
    related = dict(related)
    related = {str(key): str(value) for key, value in related.items()}
    return json.dumps(related)

