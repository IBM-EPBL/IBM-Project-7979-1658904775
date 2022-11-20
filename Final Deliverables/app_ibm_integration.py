from flask import Flask, request, jsonify, render_template, url_for , request
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import requests

API_KEY = "rwsCVupJNjytOKgOkhLD3zUweTkAKg7329eJPSqU3_F4"

token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})


mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


# Import dataset 
df = pd.read_csv('Data/Processed_data15.csv')

# Label Encoding
le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])

le_dest = LabelEncoder()
df['dest'] = le_dest.fit_transform(df['dest'])

le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

# Converting Pandas DataFrame into a Numpy array
X = df.iloc[:, 0:6].values # from column(years) to column(distance)
y = df['delayed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=61) # 75% training and 25% test

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def call_model(x):
    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    x = x[0]
    arr = [x[0], x[1], x[2], x[3], x[4], x[5]]
    a0 = int(x[0])
    a1 = int(x[1])
    a2 = int(x[2])
    a3 = int(x[3])
    a4 = int(x[4])
    a5 = int(x[5])

    #payload_scoring = {"input_data": [{"fields": ['year', 'month', 'day', 'carrier', 'origin','dest'], "values": [[2023, 12, 1, 15,2,134]]}]}
    
    payload_scoring = {"input_data": [{"fields": ['year', 'month', 'day', 'carrier', 'origin','dest'], "values": [[a0,a1,a2,a3,a4,a5]]}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/db3ecc46-7554-4c39-8ace-8b5373b0dc80/predictions?version=2022-11-19', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    ans = response_scoring.json()
    print(ans)
    ans = ans['predictions'][0]['values'][0][1][0]*100
    if ans > 85:
        ans = 1
    else:
        ans = 0
    return ans


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    year = request.form['year']
    month = request.form['month']
    day = request.form['day']
    carrier = request.form['carrier']
    origin = request.form['origin']
    dest = request.form['dest']
    year = int(year)
    month = int(month)
    day = int(day)
    carrier = str(carrier)
    origin = str(origin)
    dest = str(dest)
    
    if year >= 2013:
        x1 = [year,month,day]
        x2 = [carrier, origin, dest]
        x1.extend(x2)
        df1 = pd.DataFrame(data = [x1], columns = ['year', 'month', 'date', 'carrier', 'origin', 'dest'])
        
        df1['carrier'] = le_carrier.transform(df1['carrier'])
        df1['origin'] = le_origin.transform(df1['origin'])
        df1['dest'] = le_dest.transform(df1['dest'])
 
        x = df1.iloc[:, :6].values
        ans = call_model(x)
        output = ans
    
    return render_template('index.html', prediction_text=output)



if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)   
    app.run(debug=False) 
#if __name__ == '__main__':
#	app.run(debug=False)
# For mac, make 'app.run(debug=True)'
