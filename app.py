import openai
import spacy
from textblob import TextBlob
from translate import translate

def chat_with_me(prompt, context):
    openai.api_key = 'YOUR_API_KEY'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop=None
    )
    return response.choices[0].text.strip(), context

def how_do_u_feel(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def extract_named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def translate_text(text, target_language="en"):
    translator = translate()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def main():
    print("ถามอะไรตอบได้")
    user_history = []
    translation_mode = False
    summarization_mode = False

    while True:
        user_input = input("คุณ: ")

        if user_input.lower() in ['bye']:
            print("Bye!")
            break

        try:
            if "translate" in user_input.lower():
                translation_mode = True
                summarization_mode = False
                print("Translation mode activated.")
            elif "summarize" in user_input.lower():
                summarization_mode = True
                translation_mode = False
                print("Summarization mode activated.")

            prompt = f"User: {user_input}\nAssistant:"
            full_context = "\n".join(user_history + [prompt])
            assistant_response, updated_context = chat_with_me(full_context, full_context)
            sentiment = how_do_u_feel(assistant_response)
            entities = extract_named_entities(assistant_response)
            print(f"Assistant (Sentiment: {sentiment}, Entities: {entities}):", assistant_response)
            user_history.append(updated_context)

            if translation_mode:
                translated_response = translate_text(assistant_response, target_language="fr")
                print(f"Translated Response: {translated_response}")
            elif summarization_mode:
                summarized_text = assistant_response[:100]
                print(f"Summarized Response: {summarized_text}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
