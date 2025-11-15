import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from django.conf import settings
from django.core.wsgi import get_wsgi_application
from django.urls import path
from django.http import HttpResponse
from django.template import Template, Context
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib
matplotlib.use('Agg') # <-- ADD THIS
import matplotlib.pyplot as plt

BASE_DIR = os.getcwd()

settings.configure(
    DEBUG=True,
    SECRET_KEY="abc123",
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=["*"],
    INSTALLED_APPS=[],
    TEMPLATES=[{
        "BACKEND": "django.template.backends.django.DjangoTemplates",
    }]
)

application = get_wsgi_application()


model = joblib.load(r"C:/Users\ASUS\Documents/app/ens_model.joblib")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Churn Predictor</title>
</head>
<body>
<h1>Upload CSV</h1>

<form method="POST" enctype="multipart/form-data">
    <input type="file" name="file">
    <button type="submit">Upload & Predict</button>
</form>

{% if error %}
    <h2 style="color: red;">Error</h2>
    <p>{{error}}</p>
{% endif %}

{% if results %}
    <h2>Results</h2>
    <p><b>Accuracy:</b> {{accuracy}}</p>
    <p><b>F1 Score:</b> {{f1}}</p>

    <h3>Confusion Matrix</h3>
    <pre>{{confusion}}</pre>

    <h3>ROC Curve</h3>
    <img src="data:image/png;base64,{{roc_curve}}">
    
    <h3>Download Predictions</h3>
    <a href="/download">Download predictions.csv</a>
{% endif %}

</body>
</html>
"""

predicted_file = None

def index(request):
    """
    Main view to handle file uploads, preprocessing, prediction, and results display.
    """
    global predicted_file

    if request.method == "POST":
        uploaded = request.FILES["file"]
        
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            ctx = Context({"error": f"Failed to read CSV: {e}"})
            return HttpResponse(Template(HTML).render(ctx))
        
        df_original = df.copy() 
        
        df_processed = df.copy()

        try:
            le = LabelEncoder()
            df_processed['Gender'] = le.fit_transform(df_processed['Gender'])
            df_processed['Subscription Type'] = le.fit_transform(df_processed['Subscription Type'])
            df_processed['Contract Length'] = le.fit_transform(df_processed['Contract Length'])
        except KeyError as e:
            ctx = Context({"error": f"CSV is missing required column: {e}"})
            return HttpResponse(Template(HTML).render(ctx))
        except Exception as e:
            ctx = Context({"error": f"Error during encoding: {e}"})
            return HttpResponse(Template(HTML).render(ctx))

        required_order = [
            'Payment Delay', 'Support Calls', 'Tenure', 'Age',
            'Last Interaction', 'Subscription Type', 'Contract Length',
            'Total Spend', 'Usage Frequency', 'Gender'
        ]

        try:
            df_processed = df_processed[required_order]
        except KeyError as e:
            ctx = Context({"error": f"CSV is missing required column: {e}"})
            return HttpResponse(Template(HTML).render(ctx))
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_processed)
        
        pred = model.predict(df_scaled)

        df_original["Predicted_Churn"] = pred
        predicted_file = df_original.to_csv(index=False)

        accuracy = f1 = confusion = roc_img = None
        if "Churn" in df_original.columns:
            y_true = df_original["Churn"]
            y_pred = pred

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            confusion = str(confusion_matrix(y_true, y_pred)) 

            
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], 'r--') 
            plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.legend()

            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            
            
            roc_img = base64.b64encode(buf.read()).decode()
            
            buf.close()
            plt.close()

        
        ctx = Context({
            "results": True,
            "accuracy": accuracy,
            "f1": f1,
            "confusion": confusion,
            "roc_curve": roc_img,
        })
        return HttpResponse(Template(HTML).render(ctx))

   
    return HttpResponse(Template(HTML).render(Context()))

def download(request):
    """
    Handles the download request for the predictions CSV.
    """
    global predicted_file
    if predicted_file is None:
        return HttpResponse("No predictions available to download.", status=404)
        
    response = HttpResponse(predicted_file, content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response

urlpatterns = [
    path("", index),
    path("download", download),
]

if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line(["app.py", "runserver", "0.0.0.0:8000"])