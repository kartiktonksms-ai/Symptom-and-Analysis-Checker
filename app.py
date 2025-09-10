import streamlit as st
import pandas as pd
import pickle
import warnings
from textblob import TextBlob
from google import genai
import uuid

warnings.filterwarnings("ignore")

# ------------------------
# Load datasets
# ------------------------
dataset = pd.read_csv('merged_dataset.csv', encoding='utf-8')
symptom_severity = pd.read_csv('symptom_severity.csv', encoding='utf-8')

# ------------------------
# Load trained model and vectorizer
# ------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ------------------------
# Initialize Google GenAI client with your API key
# ------------------------
client = genai.Client(api_key="AIzaSyAVKwaDYg1tuYosWHevtsfflNGZjN-ozP8")

SYSTEM_PROMPT = (
    "You are a helpful healthcare chatbot specialized in providing personalized health advice, "
    "guiding the user through symptom checking, medication tracking, appointment assistance, and wellness tips. "
    "Interact naturally, ask relevant questions, and respond according to user's answers.\n\n"
)

# ------------------------
# Sidebar Navigation
# ------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Section", ["Symptom Checker", "Sentiment Analysis", "Healthcare Chatbot"])

# ------------------------
# Symptom Checker Section
# ------------------------
if app_mode == "Symptom Checker":
    st.title("🩺 AI Symptom Checker")
    st.write("Enter your symptoms separated by commas (e.g., fever, cough, headache)")
    user_input = st.text_input("Your Symptoms")
    if user_input:
        user_symptoms = [sym.strip() for sym in user_input.split(',')]
        user_symptom_text = " ".join(user_symptoms)
        X_input = vectorizer.transform([user_symptom_text])
        pred_probs = model.predict_proba(X_input)[0]
        pred_classes = model.classes_
        top3_idx = pred_probs.argsort()[-3:][::-1]
        top3_diseases = [pred_classes[i] for i in top3_idx]
        top3_probs = [pred_probs[i] for i in top3_idx]
        st.subheader("Top 3 Possible Diseases")
        top3_data = []
        for i in range(3):
            disease = top3_diseases[i]
            probability = top3_probs[i]
            desc_row = dataset[dataset['Disease'] == disease]
            description = desc_row['Symptom_Description'].values[0] if not desc_row.empty else "No description available"
            precautions = []
            for j in range(4):
                col_name = f"Symptom_precaution_{j}"
                if col_name in desc_row.columns and not desc_row.empty:
                    precautions.append(desc_row[col_name].values[0])
            severity_score = 0
            for sym in user_symptoms:
                if sym in symptom_severity['Symptom'].values:
                    score = symptom_severity[symptom_severity['Symptom'] == sym]['Symptom_severity'].values[0]
                    severity_score += score
            if severity_score > 15:
                st.warning(f"{i+1}. {disease} — High severity ⚠ Confidence: {probability*100:.2f}%")
            elif severity_score > 7:
                st.info(f"{i+1}. {disease} — Medium severity ℹ Confidence: {probability*100:.2f}%")
            else:
                st.success(f"{i+1}. {disease} — Low severity ✅ Confidence: {probability*100:.2f}%")
            st.markdown(f"- *Description:* {description}")
            st.markdown(f"- *Precautions:* {', '.join([p for p in precautions if p])}")
            st.markdown(f"- *Severity Score:* {severity_score}")
            st.write("---")
            top3_data.append({
                "Disease": disease,
                "Confidence": round(probability*100,2),
                "Severity Score": severity_score,
                "Description": description,
                "Precautions": ', '.join([p for p in precautions if p])
            })
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'Input Symptoms': ", ".join(user_symptoms),
            'Top Disease': top3_diseases[0],
            'Confidence': round(top3_probs[0]*100,2),
            'Severity Score': sum([top3_data[k]['Severity Score'] for k in range(3)])
        })
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(history_df)
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name='prediction_history.csv',
            mime='text/csv'
        )
        st.subheader("Top 3 Predictions Confidence Chart")
        chart_df = pd.DataFrame(top3_data)
        chart_df['Color'] = ['#ff4b4b' if x>15 else '#ffa500' if x>7 else '#2ecc71' for x in chart_df['Severity Score']]
        st.bar_chart(chart_df.set_index('Disease')['Confidence'])

# ------------------------
# Sentiment Analysis Section
# ------------------------
elif app_mode == "Sentiment Analysis":
    st.title("💬 Sentiment Analysis on Feedback")
    feedback = st.text_area("Enter your comment / feedback")
    if 'feedback_history' not in st.session_state:
        st.session_state['feedback_history'] = []
    if st.button("Analyze"):
        if feedback.strip() != "":
            analysis = TextBlob(feedback)
            sentiment_score = analysis.sentiment.polarity
            if sentiment_score > 0:
                sentiment = "Positive 😀"
            elif sentiment_score < 0:
                sentiment = "Negative 😡"
            else:
                sentiment = "Neutral 😐"
            st.write(f"*Sentiment:* {sentiment} (Score: {sentiment_score:.2f})")
            st.session_state['feedback_history'].append({
                "Feedback": feedback,
                "Sentiment": sentiment,
                "Score": round(sentiment_score, 2)
            })
    if st.session_state['feedback_history']:
        st.subheader("Feedback History")
        feedback_df = pd.DataFrame(st.session_state['feedback_history'])
        st.dataframe(feedback_df)
        csv_feedback = feedback_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Feedback History as CSV",
            data=csv_feedback,
            file_name='feedback_history.csv',
            mime='text/csv'
        )

# ------------------------
# Healthcare Chatbot Section with instant response display
# ------------------------
elif app_mode == "Healthcare Chatbot":
    st.title("🤖 Healthcare Chatbot")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_input = st.text_input("Enter your health question or message", key="chat_input")
    send = st.button("Send")

    # Local copy for display including new messages this run
    display_history = st.session_state['chat_history'].copy()

    if send and user_input.strip():
        # Append user message immediately
        display_history.append({'role': 'user', 'content': user_input})

        conversation_text = SYSTEM_PROMPT
        for msg in display_history:
            prefix = "User: " if msg['role'] == 'user' else "Assistant: "
            conversation_text += prefix + msg['content'] + "\n"
        conversation_text += "Assistant:"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=conversation_text
            )
            bot_response = response.text.strip()
        except Exception as e:
            bot_response = f"Sorry, an error occurred: {str(e)}"

        # Append bot response immediately for display
        display_history.append({'role': 'bot', 'content': bot_response})

        # Update session state so history persists
        st.session_state['chat_history'] = display_history

    # Display chat (including new user and bot messages instantly)
    for message in display_history:
        if message['role'] == 'user':
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")
