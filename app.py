from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            age=float(request.form.get('age')),
            workclass= str(request.form.get('workclass')),
            fnlwgt= int(request.form.get('fnlwgt')),
            education= str(request.form.get('education')),
            education_num= int(request.form.get('education_num')),
            marital_status= str(request.form.get('marital_status')),
            occupation= str(request.form.get('occupation')),
            relationship= str(request.form.get('relationship')),
            race= str(request.form.get('race')),
            sex= str(request.form.get('sex')),
            capital_gain= str(request.form.get('capital_gain')),
            capital_loss= str(request.form.get('capital_loss')),
            hours_per_week= str(request.form.get('hours_per_week')),
            native_country= str(request.form.get('native_country'))

        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        if results !=0.:
            results ="income is >=50k"
        else:
            results="income is <=50k"

        return render_template('result.html',final_result=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)