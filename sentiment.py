import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from sklearn.metrics import classification_report
from datasets import load_dataset
import matplotlib.pyplot as plt
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = load_dataset("imdb", split="test[:1000]")

def preprocess(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model.eval()
all_preds = []
all_labels = []

for batch in torch.utils.data.DataLoader(encoded_dataset, batch_size=16):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=3))

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label}, Confidence: {score:.2f}"

example_text =input("enter the text you want to convey")
print(analyze_sentiment(example_text))

def plot_confidences(texts):
    results = sentiment_pipeline(texts)
    labels = [r['label'] for r in results]
    scores = [r['score'] for r in results]
    
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(texts)), scores, color='skyblue')
    plt.yticks(range(len(texts)), texts)
    plt.xlabel("Confidence Score")
    plt.title("Sentiment Confidence")
    plt.tight_layout()
    plt.show()

# Test plot
plot_confidences(example_text)
