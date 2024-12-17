# 캡스톤디자인 그로쓰 05팀 그린나래 AI(감정분석 서버) 레포

### Members😄
- 프로젝트 기간: 24.03~24.12
- 김여은: FE(리더), BE
- 우정아: BE(리더), AI
- 장서연: AI(리더), FE
<br><br>

## 프로젝트 소개📂
✔️ 서비스명: 소담 - 소리로 담는 나만의 작은 이야기

✔️ 주제: 노인 인지 기능 저하 예방을 위한 GPT-4o 기반 음성 챗봇 및 일기 생성 서비스

✔️ 부제: 대화로 기억을 지키는 치매 예방 솔루션

<br><br>

## 기능 소개📂
✔️ 기능1: AI와의 음성 채팅<br>
: 사용자가 설정한 시간에(ex. 저녁 7시) 규칙적으로 하루에 있었던 일, 수면 시간, 섭취 음식, 약 복용 여부 등 여러 질문들을 챗봇이 음성으로 제공

✔️ 기능2: 일기 생성<br>
: 챗봇과 나눈 대화를 바탕으로 요약된 일기 생성

✔️ 기능3: report 제공<br>
: 챗봇과 나눈 대화를 바탕으로 report 제공.<br>
: 리포트에는 감정 분석 결과, 컨디션, 기억점수 그래프가 포함<br>

✔️ 기능4: 기억점수 측정 및 치매 자가진단<br>
: 챗봇과의 음성 대화를 통해 3일간의 일기 데이터를 바탕으로 기억점수 측정<br>
: TTS 기능을 활용한 자가진단(KDSQ, PRMQ)을 통해 추가적인 인지기능 저하 체크<br>


노인 사용자/ 보호자 사용자의 기능 비교<br>

![image](https://github.com/user-attachments/assets/040638f8-0479-4595-9aad-38cc014b6a94)

<br><br>

## 감정분석모델 사용 기술 및 데이터💻
- https://github.com/SKTBrain/KoBERT에서 불러온 모델이므로, 코드 실행 시 각 라이브러리 버전 확인 필요
- 사용 데이터: 감성 대화 말뭉치(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86) & 감정 분류를 위한 대화 음성 데이터(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263)
- Flask

<br><br>
## Source 코드 설명
- ```모델 생성```: https://github.com/EWHA-DraWings/Submit/blob/main/%EC%86%8C%EB%8B%B4_%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D.ipynb을 참고하세요.
- ```app.py```: input으로 들어온 일기를 문장 단위로 나누어 각 문장에 대해 감정을 분석하고 비율을 계산합니다. 이후 상위 3개의 감정과 비율을 반환합니다.
<br><br>

## How to build & Install🔨
1. Ubuntu 서버 접속
AWS에서 EC2 인스턴스를 생성한 후, 아래 명령어를 통해 서버에 접속하기

```
ssh -i /path/to/your-key.pem ubuntu@your-ec2-public-ip
```

2. Flask 및 필요 패키지 설치
- KoBERT의 용량이 커서 메모리 부족 문제가 발생할 수 있음
  - EBS 용량을 15로 늘려주기
  - Swap 파일 생성 및 설정
    ```
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    ```
   
```
sudo apt update # 시스템 업데이트
sudo apt install python3-pip # pip 설치
pip3 install Flask torch transformers nltk kobert_tokenizer
```

- KoBERT 설치하는데 오류가 발생한다면 필요한 모듈을 직접 다운로드해주기
```
# 저장소 클론
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT/kobert_hf

# 수동 설치
pip install .
```

3. EC2에 모델 올리기
   사전에 학습시켜놓은 모델을 EC2로 옮겨주기
   ```
   scp -i /path/to/model.pt ubuntu@<EC2_IP>:/home/ubuntu/emotion_analysis/
   ```

4. 소스 코드 클론
프로젝트 소스 코드를 clone해오기
```
git clone https://github.com/EWHA-DraWings/BE.git
git pull origin main # 최신 코드 pull
```

5. app 실행
```
python3 app.py
```

<br><br>
## How to test📜
1. APK 테스트는 https://github.com/EWHA-DraWings/FE의 README를 참고해주세요.
2. 로컬 플라스크 서버 실행 후, http://localhost:5000/predict 를 End Point로 감정 분석 요청하는 방식으로 테스트 가능합니다.
   ![image](https://github.com/user-attachments/assets/66844581-91c4-45e4-b8af-58c4c639feab)

