import requests
import json

req_rec = requests.get("http://127.0.0.1:5000/recommend", params={'user_id': 35, 'num': 40})
req_rel = requests.get("http://127.0.0.1:5000/related", params={'assignment_id': 30, 'num': 30})

print(req_rec.status_code, req_rec.reason)
print(req_rel.status_code, req_rel.reason)

# json_data_rec = json.loads(req_rec.text)
# json_data_rel = json.loads(req_rel.text)

# print(json_data_rec)
# print(json_data_rel)