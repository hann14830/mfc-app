import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. 같은 폴더에 있는 요리 노트(데이터) 불러오기
data = pd.read_csv('mfc_experiment_data.csv')

# 2. 전압 예측에 필요 없는 '실험번호'와 모두 동일한 '전극' 데이터 삭제
data = data.drop(columns=['실험번호', '전극'])

# 3. 글자로 된 '첨가조건'을 AI가 계산할 수 있게 변환
data = pd.get_dummies(data)

# 4. 원인(조건, 시간)과 결과(전압) 나누기
X = data.drop('측정전압(mV)', axis=1) # 원인
y = data['측정전압(mV)']              # 정답 (결과)

# 5. AI 주방장 부르기 (RandomForest)
model = RandomForestRegressor(random_state=42)

# 6. 학습 시작!
model.fit(X, y)
print("🎉 AI 학습 완료! VS Code에서 머신러닝 모델이 성공적으로 돌아갔습니다.")

# --- (기존 코드 아래에 이어서 붙여넣기) ---

print("\n--- 🤖 AI 예측 테스트 ---")

# 1. AI에게 물어볼 새로운 조건 세팅 (예: 갯벌 세균 배양액을 넣고 5분 뒤)
new_experiment = pd.DataFrame({
    '측정시간(분)': [5],
    '첨가조건': ['갯벌 세균 배양액']
})

# 2. AI가 알아들을 수 있게 데이터 모양 맞춰주기
new_experiment_dummy = pd.get_dummies(new_experiment)
new_experiment_dummy = new_experiment_dummy.reindex(columns=X.columns, fill_value=0)

# 3. 전압 예측하기!
predicted_voltage = model.predict(new_experiment_dummy)

# 4. 결과 출력
print(f"💡 AI 예측 결과: 갯벌 세균 배양액 조건에서 5분 뒤 예상 전압은 약 {predicted_voltage[0]:.1f} mV 입니다!")