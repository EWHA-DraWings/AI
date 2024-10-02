import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import nltk
from collections import Counter
from flask_cors import CORS

nltk.download('punkt')

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooler = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = lambda x: bert_tokenizer(x, max_length=max_len, padding='max_length' if pad else 'do_not_pad', truncation=True, return_tensors='pt')
        self.sentences = [transform(i[sent_idx]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i]['input_ids'].squeeze(), self.sentences[i]['token_type_ids'].squeeze(), self.sentences[i]['attention_mask'].squeeze(), self.labels[i]

    def __len__(self):
        return len(self.labels)

emotion_labels = ["불안", "당황", "분노", "슬픔", "일상", "행복", "혐오"]

app = Flask(__name__)
CORS(app)  # 모든 도메인에서 요청 허용

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bert = BertModel.from_pretrained('skt/kobert-base-v1')
model = torch.load('C:/Users/kjjgb/졸프_감정분석모델/best_model.pt', map_location=torch.device('cpu'))
model.eval()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs
    probabilities = softmax(logits.numpy())
    return probabilities[0]

def analyze_emotions(diary):
    sentences = nltk.sent_tokenize(diary)
    emotion_counts = Counter()
    non_neutral_sentences = 0

    for sentence in sentences:
        probabilities = predict(sentence)
        top_emotion_index = np.argmax(probabilities)

        if top_emotion_index != 4:
            emotion_counts[top_emotion_index] += 1
            non_neutral_sentences += 1

    emotion_percentages = {emotion_labels[i]: (count / non_neutral_sentences) * 100 for i, count in emotion_counts.items()}
    top_3_emotions = dict(Counter(emotion_percentages).most_common(3))

    return top_3_emotions

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    diary = data['diary']
    top_3_emotions = analyze_emotions(diary)
    return jsonify(top_3_emotions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
