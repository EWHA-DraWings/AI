import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import nltk
from collections import Counter

# NLTK punkt 데이터 다운로드
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

# 감정 레이블 정의
emotion_labels = ["불안", "당황", "분노", "슬픔", "일상", "행복", "혐오"]

app = Flask(__name__)

# 모델 및 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bert = BertModel.from_pretrained('skt/kobert-base-v1')
model = torch.load('C:/Users/kjjgb/졸프_감정분석모델/best_model.pt', map_location=torch.device('cpu'))
model.eval()

# 소프트맥스 함수 정의
def softmax(x):
    e_x = np.exp(x - np.max(x))  # 오버플로우 방지를 위해 입력에서 최대값을 뺌
    return e_x / e_x.sum(axis=1, keepdims=True)  # 각 샘플에 대해 소프트맥스 계산

# 문장별 감정 예측 함수
def predict(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs
    probabilities = softmax(logits.numpy())
    return probabilities[0]

# 일기 전체에서 감정 추출 및 분석
def analyze_emotions(diary):
    sentences = nltk.sent_tokenize(diary)  # 문장 단위로 나누기
    emotion_counts = Counter()  # 감정 빈도수 저장할 Counter 객체 생성
    non_neutral_sentences = 0  # "일상"을 제외한 문장 수 카운트

    for sentence in sentences:
        probabilities = predict(sentence)  # 각 문장에 대해 감정 예측
        top_emotion_index = np.argmax(probabilities)  # 가장 높은 확률의 감정 인덱스

        # "일상"을 중립 감정으로 간주하고 제외
        if top_emotion_index != 4:  # 4는 "일상"에 해당
            emotion_counts[top_emotion_index] += 1  # 해당 감정 카운트 증가
            non_neutral_sentences += 1  # 중립이 아닌 문장 수 증가

    # 감정 비율 계산
    emotion_percentages = {emotion_labels[i]: (count / non_neutral_sentences) * 100 for i, count in emotion_counts.items()}

    # 상위 3개 감정 추출
    top_3_emotions = dict(Counter(emotion_percentages).most_common(3))

    return top_3_emotions  # 상위 3개 감정과 그 비율 반환

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    diary = data['diary']
    top_3_emotions = analyze_emotions(diary)
    return jsonify(top_3_emotions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
