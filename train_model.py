
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

drug_pairs = [
    'aspirin warfarin', 'ibuprofen alcohol', 'acetaminophen vitaminC', 'amoxicillin ibuprofen',
    'aspirin ginkgo', 'warfarin vitaminK', 'metformin insulin', 'acetaminophen ibuprofen',
    'atorvastatin grapefruit', 'clopidogrel omeprazole', 'digoxin verapamil', 'lisinopril potassium',
    'sildenafil nitrates', 'codeine alcohol', 'diazepam alcohol', 'alprazolam ketoconazole',
    'levothyroxine calcium', 'ciprofloxacin antacids', 'methotrexate NSAIDs', 'tramadol SSRIs',
    'phenytoin folic acid', 'fluoxetine tramadol', 'doxycycline dairy', 'nitroglycerin sildenafil',
    'spironolactone potassium', 'clonidine beta-blockers', 'acetaminophen hydrocodone',
    'amoxicillin clavulanate', 'aspirin acetaminophen', 'loratadine pseudoephedrine'
]

interaction_labels = [
    1, 1, 0, 0,
    1, 1, 0, 0,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 0, 0, 0,
    0, 0
]

df = pd.DataFrame({
    'drug_pair': drug_pairs,
    'interaction': interaction_labels
})

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(df['drug_pair'], df['interaction'])

joblib.dump(model, "drug_interaction_model.pkl")
print("✅ Model trained and saved as 'drug_interaction_model.pkl'")
