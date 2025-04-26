from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

# ── Expanded Training Data ──────────────────────────────────────────────────
# Original pairs + additional real-world interactions
drug_pairs = [
    'aspirin warfarin', 'ibuprofen alcohol', 'acetaminophen vitaminC', 'amoxicillin ibuprofen',
    'aspirin ginkgo', 'warfarin vitaminK', 'metformin insulin', 'acetaminophen ibuprofen',
    'atorvastatin grapefruit', 'clopidogrel omeprazole', 'digoxin verapamil', 'lisinopril potassium',
    'sildenafil nitrates', 'codeine alcohol', 'diazepam alcohol', 'alprazolam ketoconazole',
    'levothyroxine calcium', 'ciprofloxacin antacids', 'methotrexate NSAIDs', 'tramadol SSRIs',
    'phenytoin folic acid', 'fluoxetine tramadol', 'doxycycline dairy', 'nitroglycerin sildenafil',
    'spironolactone potassium', 'clonidine beta-blockers', 'acetaminophen hydrocodone',
    'amoxicillin clavulanate', 'aspirin acetaminophen', 'loratadine pseudoephedrine',
    'ibuprofen aspirin', 'paracetamol caffeine', 'metformin aspirin', 'atorvastatin amlodipine',
    'omeprazole clopidogrel', 'warfarin ibuprofen', 'aspirin clopidogrel', 'metoprolol amlodipine',
    'tramadol acetaminophen', 'prednisone ibuprofen', 'amoxicillin azithromycin', 'lisinopril aspirin',
    'atorvastatin warfarin', 'metformin glipizide', 'hydrochlorothiazide lisinopril',
    'gabapentin hydrocodone', 'oxycodone acetaminophen', 'naproxen ibuprofen', 'aspirin naproxen',
    'ibuprofen diclofenac', 'paracetamol ibuprofen', 'amoxicillin doxycycline',
    'ciprofloxacin metronidazole', 'levothyroxine iron', 'fluoxetine sertraline',
    'diazepam lorazepam', 'clonazepam alprazolam', 'sildenafil tadalafil',
    'metformin sitagliptin', 'atorvastatin ezetimibe',
    'azithromycin ciprofloxacin', 'fluconazole warfarin', 'loratadine metoprolol',
    'pantoprazole clopidogrel', 'venlafaxine tramadol', 'zolpidem diazepam',
    'levetiracetam carbamazepine', 'allopurinol warfarin', 'propranolol metoprolol',
    'clonidine atenolol', 'pioglitazone insulin', 'rivaroxaban aspirin',
    'dabigatran warfarin', 'esomeprazole clopidogrel', 'lansoprazole warfarin',
    'itraconazole simvastatin', 'lamotrigine valproate', 'topiramate acetazolamide',
    'eszopiclone zolpidem', 'vilazodone SSRIs', 'bupropion SSRIs',
    'duloxetine SSRIs', 'lithium NSAIDs', 'buprenorphine methadone',
    'naloxone buprenorphine', 'methadone naloxone'
]

# 0 = no interaction, 1 = interaction
interaction_labels = [
    1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,1,1,
    0,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,
    0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,0,0
]

#Check if the lengths match
if len(drug_pairs) != len(interaction_labels):
    raise ValueError(f"Mismatch: {len(drug_pairs)} pairs vs {len(interaction_labels)} labels")

#prepare the data
df = pd.DataFrame({
    'drug_pair': drug_pairs,
    'interaction': interaction_labels
})
#sklearn for model training
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(df['drug_pair'], df['interaction'])

#save the model
joblib.dump(model, "drug_interaction_model.pkl")
print("Model trained and saved as 'drug_interaction_model.pkl'")
