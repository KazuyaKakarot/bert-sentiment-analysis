from  datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import pandas as pd

print("Loading IMDb's dataset...")
dataset = load_dataset("imdb")

print(dataset)

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print(f"Train size: {len(train_texts)} | Test size: {len(test_texts)}")


#TF-IDF Vectorizer
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=20000,ngram_range=(1,2),stop_words="english")
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

#Logistic Regression baseline
print("Training baseline model...")
model = LogisticRegression(max_iter=1000,C=1.0)
model.fit(X_train,train_labels)

#Evaluate
preds = model.predict(X_test)
f1 = f1_score(test_labels,preds)

print("\n=== Baseline Results ===")
print(classification_report(test_labels,preds,target_names=["Positive","Negative"]))
print(f"F1 Score: {f1:.4f}")

# Save results for later comparision

results = pd.DataFrame({"Model":["TF-IDF + Logistic Regression"],"F1 score":[round(f1,4)]})
results.to_csv("data/baseline_results.csv",index=False)
print("Results saved to data/baseline_results.csv")





