import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "Training.csv"
MODEL_PATH = "model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"


def main():
    data = pd.read_csv(DATA_PATH)
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    symptoms = [col for col in data.columns if col != "prognosis"]
    X = data[symptoms]
    y = data["prognosis"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"Model trained with validation accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
