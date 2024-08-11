from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scaler
model_dalc = joblib.load('model_dalc.pkl')
model_walc = joblib.load('model_walc.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            features = [
                request.form.get('school', type=int),
                request.form.get('sex', type=int),
                request.form.get('age', type=int),
                request.form.get('address', type=int),
                request.form.get('famsize', type=int),
                request.form.get('Pstatus', type=int),
                request.form.get('Medu', type=int),
                request.form.get('Fedu', type=int),
                request.form.get('Mjob', type=int),
                request.form.get('Fjob', type=int),
                request.form.get('reason', type=int),
                request.form.get('guardian', type=int),
                request.form.get('traveltime', type=int),
                request.form.get('studytime', type=int),
                request.form.get('failures', type=int),
                request.form.get('schoolsup', type=int),
                request.form.get('famsup', type=int),
                request.form.get('paid', type=int),
                request.form.get('activities', type=int),
                request.form.get('nursery', type=int),
                request.form.get('higher', type=int),
                request.form.get('internet', type=int),
                request.form.get('romantic', type=int),
                request.form.get('famrel', type=int),
                request.form.get('freetime', type=int),
                request.form.get('goout', type=int),
                request.form.get('Dalc', type=int),
                request.form.get('Walc', type=int),
                request.form.get('health', type=int),
                request.form.get('absences', type=int),
                request.form.get('G1', type=int)
            ]

            # Convert to numpy array and scale
            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)

            # Make predictions
            prob_dalc = model_dalc.predict_proba(features_scaled)[0][1] * 100
            prob_walc = model_walc.predict_proba(features_scaled)[0][1] * 100

            # Calculate general probability and round it
            prob_general = round((prob_dalc + prob_walc) / 2, 2)

            return render_template('form.html', prediction_general=prob_general)

        except Exception as e:
            print("Error during prediction:", e)
            return render_template('form.html', prediction_general=None, error=str(e))

    return render_template('form.html', prediction_general=None)

if __name__ == '__main__':
    app.run(debug=True)
