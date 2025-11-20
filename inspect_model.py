# inspect_model.py
import joblib
obj = joblib.load("rf_health_model.joblib")

print("Top keys in saved object:", list(obj.keys()))
if "feature_columns" in obj:
    print("\nFeature columns (count={}):".format(len(obj["feature_columns"])))
    print(obj["feature_columns"])
else:
    print("\nNo 'feature_columns' key found.")

# pipeline or model
if "pipeline" in obj:
    pipe = obj["pipeline"]
    print("\nPipeline steps:", pipe.named_steps.keys())
    # classifier classes (if exists)
    try:
        clf = pipe.named_steps[list(pipe.named_steps.keys())[-1]]
        print("\nClassifier type:", type(clf))
        print("Classifier classes_:", getattr(clf, "classes_", "N/A"))
    except Exception as e:
        print("Could not get classifier details:", e)
else:
    print("\nNo 'pipeline' key; try keys above.")
