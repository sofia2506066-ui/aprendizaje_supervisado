from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "titanic_model.joblib"
app = Flask(__name__)
models = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = request.form.to_dict()

    if request.method == "POST":
        try:
            passenger = pd.DataFrame(
                [
                    {
                        "Pclass": int(request.form["Pclass"]),
                        "Sex": request.form["Sex"],
                        "Age": float(request.form["Age"]),
                        "SibSp": int(request.form["SibSp"]),
                        "Parch": int(request.form["Parch"]),
                        "Fare": float(request.form["Fare"]),
                        "Embarked": request.form["Embarked"],
                    }
                ]
            )
            model_name = request.form["model"]
            model = models[model_name]
            survived = int(model.predict(passenger)[0])
            probability = float(model.predict_proba(passenger)[0][1])
            prediction = {
                "model": model_name,
                "label": "Sobreviviria" if survived else "No sobreviviria",
                "probability": f"{probability:.1%}",
            }
        except (KeyError, TypeError, ValueError) as exc:
            error = f"Revisa los valores introducidos: {exc}"

    return render_template(
        "index.html", prediction=prediction, error=error, form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)