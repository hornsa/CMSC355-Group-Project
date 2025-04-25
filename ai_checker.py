import joblib
from gui import run_gui

#Run the GUI in a separate thread
if __name__ == "__main__":
    run_gui()
try:
    model = joblib.load("drug_interaction_model.pkl")
except FileNotFoundError:
    raise FileNotFoundError("File not found. Make sure drug_interaction_model.pkl exists.")
def predict_interaction(primary_drug, second_drug):
#Lowercase input
    drug_pair = f"{primary_drug.lower()} {second_drug.lower()}"
#Use the AI model to make a prediction based on the pairs
    prediction = model.predict([drug_pair])
    probabilities = model.predict_proba([drug_pair])[0]
    confidence = max(probabilities)
#Use the confidence level from ^ to help give an output for the program
    if confidence < 0.6:
        return "This combination is unfamiliar. Please verify with a professional."
    elif prediction[0] == 1:
        return "Potential interaction detected!"
    else:
        return "No known interaction detected."
def is_input_valid (prompt):
    while True:
        user_input = input(prompt)
        if user_input.strip():
            return user_input
        else:
            print("Invalid input. Please enter a valid drug name.")

print("Drug Interaction Checker")
while True: 
    primary_drug = is_input_valid("Enter the first drug: ")
    second_drug =  is_input_valid("Enter the second drug: ")
    print("You entered: " + primary_drug + " and " + second_drug)
    correctinput = input("Are these inputs correct? (yes/no): ").strip().lower()
    if correctinput == "yes":
        break
    else:
     print("Please try again")

result = predict_interaction(primary_drug, second_drug)
print("Drugs that are being tested " + primary_drug + " and " + second_drug)
print(result)


    
