import json
import numpy as np

file = open('Data/mapping case id - sample UUID.json')
mapper = json.load(file)

print('Sample UUID:', mapper[0]['file_name'].split('.')[0])
print('Case ID', mapper[0]['cases'][0]['case_id'])
# for case in mapper[0]['cases']:
#     print('Case ID', case['case_id'])


s = 0
for item in mapper:
    if len(item['cases']) > 1:
        s += 1
print(s, 'sample with more than one case exist.')
