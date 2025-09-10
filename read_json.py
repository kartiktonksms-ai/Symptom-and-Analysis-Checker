import json

with open("disease_medicine.json", "r") as f:
    disease_db = json.load(f)


user_symptoms = input("Enter your symptoms: ").lower()

matched = False
for disease, details in disease_db.items():
    if any(symptom in user_symptoms for symptom in details["Symptoms"]):
        print(f"\nPossible Disease: {disease}")
        print(f"Urgency Level: {details['Urgency']}")
        print("Suggested Medicines:", ", ".join(details["Medicines"]))
        print("⚠️ Always consult a doctor before taking medicines.")
        matched = True
        break

if not matched:
    print("No disease matched. Please consult a doctor.")
