from transformers import pipeline

# Load pre-trained model and tokenizer
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

def classify_intent(text):
    result = classifier(text)
    intent = result[0]['label'].lower()
    return intent
