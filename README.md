# AI_Engineering_Basic
**[naver connect] boostcourse - AI 엔지니어 기초 다지기**

<img src="https://github.com/thals304/AI_Engineering_Basic/assets/126967336/8c6f3134-61cc-451b-94cc-cb5489ae43c7.png" width="300" height="400"/>

## ⏱️스터디 기간

2024.01.05 ~ 2024.02.27

## ⚙️개발 환경

- anaconda3 1.12.1

- jupyter notebook 6.5.4 (Phython 3.12.1)

## 01boostclass : numpy & pandas

```python
### [TODO] 코드 구현 1 : 행렬곱 연산
##본격적으로 Numpy와 친해지기 위해서 다양한 연산을 연습해볼 예정입니다. 
##랜덤으로 무작위 데이터를 가진 5x3 행렬과 3x2 행렬을 numpy array로 만든 후, 행열곱 연산을 진행해봅시다. 
##그리고 그 결과를 출력해봅시다.
import numpy as np

## 코드시작 ##

matrix_a = np.random.rand(5,3)

matrix_b = np.random.rand(3,2)

result_matrix = np.dot(matrix_a,matrix_b)

## 코드종료 ##
print(result_matrix, result_matrix.shape)
```

- **numpy 가져오기**

```python
import numpy as np
```

- **numpy 배열 생성 : np.array()**

```python
x = np.array([1,2,3,4])
```

- **산술 연산**

```python
x = np.array([1,2,3,4])
y = np.array([5,6,7,8])

print(x+y)
print(x-y)
print(x/y)
```

- **N차원 배열**

```python
x = np.array([1,2],[3,4])
y = np.array([5,6],[7,8])

print(x.shape)    # 배열 형상 보기
print(np.ndim(x)) # 배열 차원 보
```

→ shape이 같은 행렬끼리 연산 가능

- **브로드캐스트**
    
    : shape이 다른 행렬끼리의 연산
    

```python
x = np.array([[1, 2], [3, 4]])
y = np.array([10, 20])

print(x*y)
```

- **원소 접근하기**

```python
x = np.array([[1, 2], [3, 4]])

print(x[0]) # array([1, 2])
print(x[0][1]) # 2.0
```

- **그래프 그리기**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3, 0.1) # 0부터 1까지 0.1 간격으로 생성
y = np.cos(x)            # cos 그래프

plt.plot(x, y)
plt.show()
```

- **그래프 이름과 축 추가하기**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1) # 0부터 1까지 0.1 간격으로 생성
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend() # 범례추가
plt.show()
```

- **행렬의 곱**

```python
x = np.array([[1, 2], [3, 4]])
y = np.array([[4, 5], [6, 7]])

print(np.dot(x, y))
```

- **랜덤 행렬 생성**

```python
x = np.random.rand(2,2)  # 2x2 행렬 랜덤 생성
y = np.random.rand(3,2)  # 3x3 행렬 랜덤 생성
```

```python
## [TODO] 코드 구현 2 : concatenate 연산
## 그 다음으로는 numpy에서 자주 사용하는 concatenate 연산을 해보겠습니다. 
## 다음 array를 사용해 axis 가 0과 1일때의 concatenate 연산을 각각 구해보세요.

*   첫번째 array : [[5,7], [9,11]]
*   두번째 array : [[2,4], [6,8]]

import numpy as np

## 코드 시작

x = np.array(([5,7],[9,11]))
y = np.array(([2,4],[6,8]))

## 코드 종료
print(np.concatenate((x,y),axis = 0), np.concatenate((x,y),axis = 1))
```

- **concatentate 메소드 (배열 합치기) + axis(축)**

: concatenate 메소드는 선택한 축(axis)의 방향으로 배열을 연결해주는 메소드

```python
## 1차원 배열에서 concatenate & asix
import numpy as np
x1 = np.array([1,2,3])
y1 = np.array([4,5,6])
np.concatenate((x1,y1),axis= 0)    ## asix = 0은 직선, 1은 안됨

## 2차원 배열에서 concatenate & axis
x2 = np.array([1,2,3],[10,20,30])
y2 = np.array([4,5,6],[40,50,60])
np.concatenate((x2,y2), axis = 1   ## asix = 0은 행(위->아래) 방향, 1은 열(좌->우)

## 3차원 배열
## axis = 0은 높이방향, 1은 행 방향, 2는 열 방향

```

```python
## [TODO] 코드 구현 3 : Series 코드 완성하기 - 조건에 따른 목록 재구성
## 다음의 재고 목록을 사용해 Pandas 라이브러리의 Series 형태를 만들어보세요. 
## 그리고 데이터의 10 이상 20 이하의 데이터만 골라 출력해봅시다.
<재고 목록>
* HDD : 19개
* SDD : 11개
* USB : 5개
* CLOUD : 97개
```

- **Pandas 불러오기**

```python
import pandas as pd
```

- **Series 자료구조**

```python
## 행의 이름(index), 열의 이름(name)
s = pd.Series(data, index = index, name = name)
```

- **Series 생성하기 : list & dict**

```python
## list
s = pd.Series(['KIM','SO','MIN'])

##dict
s = pd.Series({'KIM':20,'SO': 28, 'MIN': 36})
```

- **Series 속성 : index & values**

```python
s.index
s.values
```

- **Series 인덱싱**

```python
s['KIM'] = 55
s
```

- **Series 부등식(조건)**

```python
## 20보다 크고 30보다 작은 데이터
s[(s>20) & (s<30)]
```

```python
##[TODO] 코드 구현 4 : dataframe 코드 완성하기 - 야채과일 가격 계산하기
##다음과 같이 야채와 과일 목록이 정리된 데이터가 있습니다. 
##이 두 데이터를 따로 보기엔 효율성이 떨어져 1개로 합쳐 보려 합니다. 
## 각 표에 정리된 데이터를 각각 하나의 데이터 프레임으로 생성한 뒤, 하나로 결합해보세요. 
## 그리고 'type'을 이용해 데이터를 정렬하고, 가장 비싼 야채와 가장 비싼 과일의 가격 합을 구해보세요.
import pandas as pd

## 코드시작 ##
frame1 = {
    'name' : ['cherry','mango','potato','onion'],
    'type' : ['fruit','fruit','vegetable','vegetable'],
    'price' : [100,110,60,80]
}
frame1_df = pd.DataFrame(frame1)
               
frame2 = {
    'name' : ['pepper','carrot','banana','kiwi'],
    'type' : ['vegetable','vegetable','fruit','fruit'],
    'price' : [50,70,90,120]
}
frame2_df = pd.DataFrame(frame2)

combined_df = pd.concat([frame1_df, frame2_df])

sorted_df = combined_df.sort_values(by ='type')

Top_vegetable_price = sorted_df.loc[sorted_df['type'] == 'vegetable','price'].max()
Top_fruit_price = sorted_df.loc[sorted_df['type'] == 'fruit', 'price'].max()
sum_of_Top = Top_vegetable_price + Top_fruit_price

## 코드종료 ##
print(sum_of_Top)
```

- **Dataframe**

: Series들을 하나의 열로 취급한 집합 

```python
import pandas as pd
## 생성 방법(list)
frame = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]])
frame

## Dictionary 이용
data = {
      'age' : [20,23,48],
      'height' : [183,192,175],
      'weight'  : [77,83,65]
}
indexName = ['슈퍼맨','스파이더맨','배트맨']
frame = pd.DataFrame(data,index = indexName)
frame
```

- **Dataframe 조회 방법**

```python
## 열 조회
print(" 0열 조회 -1 ")
print(frame['age'])

print(frame['age'][1])   ## or print(frame.age[1]) 특정 값 조회

## 행 조회 : loc, iloc 사용
print(frame.loc['배트맨'] 
print(frmae.iloc[0])
```

```python
##[TODO] 코드 구현 5 : dataframe 코드 완성하기 - 점수 최댓값, 최솟값 출력하기
5명의 플레이어가 게임을 진행한 결과를 전달받았습니다. 총 5개의 라운드를 진행하여 각 참여자당 5개의 점수를 받았습니다. 아래에 주어진 데이터를 dataframe의 형태로 만들어 보세요. 그리고 각 라운드의 최댓값, 최솟값, 평균값을 구해 출력해봅시다.

참여자의 이름을 인덱스로 활용해보세요.
결과 출력은 dataframe의 describe() 를 활용해보세요.
<플레이어별 게임 결과>
Sue : 55, 65, 60, 66, 57
Ryan : 64, 77, 71, 79, 67
Jay : 88, 81, 79, 89, 77
Jane : 45, 35, 30, 46, 47
Anna : 91, 96, 90, 97, 99

import pandas as pd

## 코드시작 ##
data = {
    'Sue': [55, 65, 60, 66, 57],
    'Ryan': [64, 77, 71, 79, 67],
    'Jay': [88, 81, 79, 89, 77],
    'Jane': [45, 35, 30, 46, 47],
    'Anna': [91, 96, 90, 97, 99]
}
game_results_df = pd.DataFrame(data)

game_results_df = game_results_df.rename(columns={'Sue': 'round1', 'Ryan': 'round2', 'Jay': 'round3', 'Jane': 'round4', 'Anna': 'round5'})
## 코드종료 ##
print(game_results_df.describe().loc[["mean", "max", "min"]])
```

## 02boostclass : 정형데이터 & 정형데이터 분류 베이스라인 모델 1

- **정형 데이터**

엑셀 파일 형식이나 관계형 데이터베이스의 테이블에 담을 수 있는 데이터로 행(row)과 열(column)으로 표현 가능한 데이터. 하나의 행은 하나의 데이터 인스턴스를 나타내고, 각 열은 데이터 피처를 나타냄

**vs 비정형 데이터** : 이미지, 비디오, 음성, 자연어 등의 정제되지 않은 데이터

- **정형 데이터의 중요성**
    - **범용적인 데이터**
        
        사람, 기업, 현상, 사회의 많은 부분들이 정형 데이터로 기록
        
        → 가장 기본적인 데이터
        
        → 분야를 막론하고 많은 데이터가 정형 데이터로 존재
        
    - **필수적인 데이터**
    - **정형 데이터의 분석 능력**
    
   → 데이터에 대한 상상력, 통찰력 필요 (다양한 경험을 통해 데이터에 국한 되지 않고 범용적으로 쓰일 수 있는 능력)

- **데이터 및 문제 이해 (우리가 다룰 데이터)**
    
    2009년 12월부터 2011년 11월까지 온라인 상점의 거래 데이터 
    
    (데이터 행은 780,520개, 컬럼은 9개의 컬럼으로 구성)
    
   > **문제 정의**
   > 
   > X : 고객 5914명의 2009년 12월 ~ 2011년 11월까지의 구매기록
   > 
   > Y : 5914면의 2011년 12월 총 구매액 300 초과 여부 (Binary)
   > 
   > -> 타겟 마케팅, 고객 추천
   > 

    → 어떤 모델로 train, valid, test ?

- **평가지표 이해**
    - **분류(Classification)**
        
        예측해야할 대상의 개수가 정해져 있는 문제
        
        ex) 이미지에서 개, 고양이 분류 / 신용카드 거래가 사기 거래인지 정상 거래인지 분류 등
        
    - **회귀(Regression)**
        
        예측해야할 대상이 연속적인 숫자인 문제
        
        ex) 일기 예보에서 내일의 기온 예측, 주어진 데이터에서 집값 예측
        
    - **평가지표(Evaluation Metric)**
        
        분류, 회귀 머신러닝 문제의 성능을 평가할 지표
      
    +) **분류 문제** 

    **Confusion Matrix**

    1. Accuracy : (TP + TN) / (TP + TN + FP + FN)
    2. Precision : TP / (TP + FP)
    3. Recall : TP / (TP + FN)

    **ROC**

    True Positive Ratio : TP / (TP + FN)

    False Positive Ratio : FP / (FP + TN)

    **AUC** 

    ROC 곡선의 면적을 표시한 것 

    [0~1] 범위로 1에 가까우면 잘 예측한 것 0에 가까우면 잘 예측 못한 것

## 03boostclass : EDA & 정형 데이터 전처리

- **EDA 정의**
    
    **탐색적 데이터 분석**
    
    데이터를 탐색하고 가설을 세우고 증명하는 과정
    
    실제 모델링을 하기 전에 필수로 진행해야 함
    
    데이터의 특징과 내재하는 구조적 관계를 알아내기 위해 시각화와 통계적 방법을 통해 다양한 각도에서 관찰하고 이해하는 과정 
    
    → 문제를 직관적으로 이해하고, 정답에 가까워질 수 있게 됨
    
    정형데이터/비정형 데이터 구분 없이 모든 데이터 분석에서 공통적으로 진행되는 필수 과정
    
    데이터적 통찰력, 데이터적 상상력 
    
- **EDA 과정**
    
    주어진 문제를 데이터를 통해 해결하기 위해 데이터를 이해하는 과정
    
    탐색하고 생각하고 증명하는 과정의 반복
    
    데이터 마다 상이한 도메인
    
    EDA의 시작
    
    → 개별 변수의 분포
    
    → 변수간의 분포와 관계

- **EDA titanic data**
    - **데이터 파악**
    - **개별 변수**
        
        연속형, 범주형
        
    - **변수간의 관계**
 
 - **EDA our data ( 쇼핑 )**
    - **문제 이해 및 가설 세우기**
        
        가설 세우기 - 가설을 확인하면서 데이터의 특성을 파악! → 데이터 이해

- **데이터 전처리(Preprocessing)**
    
    머신러닝 모델에 데이터를 입력하기 위해 데이터를 처리하는 과정
    
    연속형, 범주형 처리
    
    결측치 처리
    
    이상치 처리
        
    - **가설 검정 - 연속형**
    - **가설 검정 - 범주형**

- **연속형, 범주형 처리**
    
    **-연속형-** 
    
    - **Scaling**
        
        데이터의 단위 혹은 분포를 변경
        
        선형기반의 모델(선형 회귀, 딥러닝 등)인 경우 변수글 간의 스케일을 맞추는 것이 필수적
        
        1. **Scaling**
        2. **Scaling + Distribution**
            
            **Binning**
            
        - Min Max scaling
        - Standard Scaling
        - Robust Scaling
    
    **-범주형-**
    
    - **Encoding**
        1. **One hot encoding** : 변수를 1과 0으로 나눔
        2. **Label encoding** : 변수마다 다른 번호 부여 / 순서의 영향을 받음
        3. **Frequency encoding** : 변수의 빈도수
        4. **Target encoding** : target 변수의 평균
            
            // 3,4 는 의미 있는 값을 줌
            
        5. **Embedding** : 낮은 차원으로 처리

- **결측치 처리**
    - **pattern**
    - **Univariate**
        - 제거
        - 평균값 삽입
        - 중위값 삽입
        - 상수값 삽입
    - **Multivariate**
        - 회귀분석
        - KNN nearest

- **이상치 처리**
    - **이상치란**
        
        데이터 중 일반적인 데이터와 크게 다른 데이터
        
    - **이상치 탐색**
        - Z-Score
        - IQR
    - **이상치 처리 관점**
        - **정성적인 측면**
            
            이상치 발생 이유
            
            이상치의 의미
            
        - **성능적인 측면**
            
            Train Test Distribution
## 04boostclass : 머신러닝 기본 개념

- **Underfitting & Overfitting**
    - **Underfitting** : 데이터를 설명하지 못함
    - **Overfitting** : 데이터를 과하게 설명함 
    full dataset = our dataset 일 때
- **Regulaization**
    - overfitting 을 규제하는 방법
        - Early stopping : 적정선 선택
        - Parameter norm penalty
        - Data augmentation : 데이터를 의도적으로 증가시킴
        - SMOTE : inbalance data를  기준으로 근처 데이터 생성
        - Dropout : feature 일부분만 사용
- **Validation strategy**
    - test dataset은 project 결과물과 직결되는 가장 중요한 set
    - validation은 내가 만들고 있는 머신러닝 모델을 test dataset에 적용하기 전에 모델의 성능을 파악하기 위해 선정하는 dataset
    - validation dataset 은 test dataset과 거의 유사하게 구성하는 것이 좋음 (test dataset은 full dataset과 유사하게 만드는 것이 좋음)
    - but, test dataset 정보를 얻을 수 없는 경우도 있음
    - training dataset은 머신러닝 모델이 보고 학습하는 dataset
- **Hold-Out Validation** : 하나의 train과 validation을 사용
    - random sampling
    - Stratified split ( 8 : 2 , 7 : 3)
- **Cross Validation**
    - 여러 개의 train과 validation을 사용
    - Stratified K-Fold ( 8 : 2 , 7 : 3)
    - Group K-Fold
    - Time series split
- **Reproducibility**
    - Fix seed
- **Machine learning workflow (머신러닝 작업 절차)**
    - 데이터 추출 후 모델링 과정 전단계
        - Data preprocessing
        - Feature scaling
        - Feature selection

## 05boostclass : 트리 모델

- **트리 모델의 기초 의사결정나무**
    - 칼럼(feature) 값들을 어떠한 기준으로 group을 나누어 목적에 맞는 의사결정을 만드는 방법
    - 하나의 질문으로 yes or no로 decision을 내려서 분류
- **트리 모델의 발전**
    
    Decision Tree → Random Forest → AdaBoost → GBM
    
- **Bagging & Boosting**
    - 여러 개의 decision tree를 이용하여 모델 생성
    - **Bagging**
        - 데이터 셋을 샘플링 하여 모델을 만들어 나가는 것이 특징
        - 샘플링한 데이터 셋을 하나로 하나의 Decision Tree가 생성
        - 생성한 Decision Tree의 Decision들을 취합하여 하나의 Decision생성
        - Bagging = Booststrap + Aggregation
        
        // Boostrap : Data를 여러 번 sampling
        
        // Aggregation : 종합(Ensemble)
        
    - **Boosting**
        - 초기의 랜덤하게 선택된 boostset를 사용하여 하나의 tree를 만들고 잘 맞추지 못한 data에 wait을 부여해 다음 tree를 만들 때 영향을 주어 다음 tree에서는 잘 맞출 수 있게 함
    
- **LightGBM, XGBoost, CatBoost**
    - XGBoost, CatBoost - 균형적 구조
    - LightGBM - 비균형적 구조
- **Tree model hyper-parameter**
    - hyper - parameter

## 06boostclass & 07boostclass : 캐글러가 되자 - housing data

> **Project**

## 08boostclass : 피처 엔지니어링

**Feature Engineering** : 원본 데이터로부터 도메인 지식 등을 바탕으로 문제를 해결하는데 도움이 되는 Feature를 생성, 변환하고 이를 머신 러닝 모델에 적합한 형식으로 변환하는 작업

## 10boostclass : 하이퍼 파라미터 튜닝

- **하이퍼 파라미터 튜닝**
    - **하이퍼 파라미터 튜닝이란?**
        - **하이퍼 파라미터** : 학습 과정에서 컨트롤 하는 파라미터 value
        - **하이퍼 파라미터 튜닝** : 하이퍼 파라미터를 최적화 하는 과정
        - **하이퍼 파라미퍼 튜닝 방법**
            - **Manual Search** : 자동화 툴을 사용하지 않고 메뉴얼하게 실험할 하이퍼 파라미터 셋을 정하고 하나씩 바꿔가면서 테스트 해보는 방식
            - **Grid Search :** 테스트 가능한  모든 하이퍼 파라미터 set을 하나씩 테스트해보면서 어떤 파라미터 set이 성능이 좋은지 기록하는 방식
            - **Random Search** : 탐색 가능한 하이퍼 파라미터를 랜덤하게 선택해 테스트 하는 방식
            - **Bayesian optimization** : 랜덤하게 하이퍼 파라미터 선택하다가 이전 성능이 잘 나온 하이퍼 파라미터를 영역을 집중적으로 탐색해서 성능이 잘 나온 구간의 하이퍼 파라미터를 선택하는 방식
    - **Boosting Tree 하이퍼 파라미터**
    - **Optuna 소개**
        - 오픈소스 하이퍼 파라미터 튜닝 프레임워크
        - **주요 기능**
            - Eager search spaces
                - Automated search for optimal hyperparameters using Python conditionals, loops, and syntax
            - State-of-the-art algorithms
                - Efficiently search large spaces and prune unpromising trials for faster results
            - Easy parallelization
                - Parallelize hyperparameter searches over multiple threads or processes without modifying code
        - Optuna 하이퍼 파라미터 탐색 결과 저장
            - storage API를 사용해서 하이퍼 파라미터 검색 결과 저장 가능
            - RDB, Redis와 같은 Persistent 저장소에 하이퍼 파라미터 탐색 결과를 저장함으로써 한 번 탐색하고, 다음에 다시 이어서 탐색 가능
        - Optuna 하이퍼 파라미터 Visualization
            - 하리퍼 파라미터 히스토리 Visualization
            - 하이퍼 파라미터 Slice Visualization
            - 하이퍼 파라미터 Contour Visualization
            - 하이퍼 파라미터 Parallel Coordinate Visualization
    - **하이퍼 파라미터 튜닝 코드 실습**

## 11boostclass : 앙상블

- **앙상블 러닝 (Ensemble learning)**
    - 여러 개의 결정 트리를 결합하여 하나의 결정 트리보다 더 좋은 성능을 내는 머신러닝 기법
    - 앙상블 학습의 핵심은 여러 개의 약 분류기(Weak Classifier)를 결합하여 강 분류기(Strong Classifier) 만드는 과정
    - 여러 개의 단일 모델들의 평균치를 내거나, 투표를 해서 다수결에 의한 결정을 하는 등 여러 모델들의 집단 지성을 활용하여 더 나은 결과를 도출해 내는 것에 주 목적이 있음
    - **장점**
        - 성능을 분산시키기 때문에 Overfitting 감소 효과
        - 개별 모델 성능이 잘 안 나올 때 앙상블 학습을 이용하면 성능이 향상될 수 있음
    - **기법**
        - **Bagging** - Boostrap Aggregation(샘플을 다양하게 생성)
            - 훈련세트에서 중복을 허용하여 샘플링하는 방식
            - +) Pasting : 중복을 허용하지 않고 샘플링하는 방식
        - **Voting(투표)** - 투표를 통해 결과 도출
            - 다른 알고리즘 model을 조합해서 사용 vs Bagging은 같은 알고리즘 내에서 다른 Sample 조합
            - Voting은 서로 다른 알고리즘이 도출해 낸 결과물에 대하여 최종 투표하는 방식
            - hard vote와 soft vote로 나뉨
            - hard vote는 결과물에 대한 최종 값을 투표해서 결정(다수결 원칙과 비슷)하고, soft vote는 최종 결과물이 나올 확률 값을 다 더해서 최종 결과물에 대한 각각의 확률을 구한 뒤 최종 값을 도출
        - **Boosting** - 이전 오차를 보완하며 가중치 부여
        - **Stacking**
            - 여러 모델들을 활용해 각각의 예측 결과를 도출한 뒤 그 예측 결과를 결합해 최종 예측 결과를 만들어 내는 것
- **Tree 계열 알고리즘 설명**
    - **Decision Tree : Impurity**
        - 해당 노드 안에서 섞여 있는 정도가 높을수록 복잡성이 높고, 덜 섞여 있을수록 복잡성이 낮음
        - Impurity를 측정하는 측도에는 다양한 종류가 있는데, Entropy와 Gini에 대해서만 알아볼 것
    - **Decision Tree :  Gini Index**
        - 불순도를 측정하는 지표로서, 데이터의 통계쩍 분산정도를 정량화해서 표현한 값
    - **Decision Tree : Graphviz**
    - **Gradient Boosting : Pseudo code**
    - **XGBoost**
        - XGBoost는 Gradient Boosting에 Regularization term을 추가한 알고리즘
        - 다양한 Loss function을 지원해 task에 따른 유연한 튜닝이 가능하다는 장점
    - **LightGBM**
        - Level-wise growth(XGBoost)의 경우 트리의 깊이를 줄이고 균형있게 만들기 위해 root 노드와 가까운 노드를 우선적으로 순회하여 수평성장하는 방법
        - leaf-wise tree growth(LighGBM)의 경우 loss 변화기 가장 큰 노드에서 분할하여 성장하는 수직 성장 방식
        - **GOSS : Gradient-based One-Side Sampling**
            - 기울기가 큰 데이터 개체 정보 획득에 있어 더욱 큰 역할을 한다는 아이디어에 입각해 만들어진 테크닉, 작은 기울기를 갖는 데이터 개체들을 일정 확률에 의해 랜덤하게 제거
        - **EFB : Exclusive Feature Bunding**
            - 변수 개수를 줄이기 위해 상호 배타적인 변수들을 묶는 기법
    - **Catboost**
        - 순서형 원칙(Ordered Principle)을 제시
            - Target leakage를 피하는 표준 그래디언트 부스팅 알고리즘을 수정하고 범주형 Feature를 처리하는 새로운 알고리즘
        - Random Permutation
            - 데이터를 셔플링하여 뽑아냄
        - 범주형 feature 처리 방법
            - ordered Target Encoding
            - Categorical Feature Combinations
            - One-Hot Encoding
        - Optimized Parameter Tuning
- **TabNet**
    - TabNet은 전처리 과정이 필요하지 않음
    - 정형 데이터에 대해서는 기존의 Decision tree-based gradient boosting(xgboost, lgbm, catboost)와 같은 모델에 비해 신경망 모델은 아직 성능이 안정적이지 못함. 두 구조의 장점을 모두 갖는 신경망 모델
    - Feature selection, interpretability(local, global)가 가능한 신경망 모델, 설명가능한 모델
    - feature 값을 예측하는 Unsupervised pretrain 단계를 적용하여 상당한 성능 향상을 보여줌
    - Tabular Data를 위한 딥러닝 모델 TabNet
        - TabNet은 순차적인 어텐션(Sequentital Attention)을 사용하여 각 의사 결정 단계에서 추론할 특징을 선택하여 학습 능력이 가장 두드러진 특징을 사용
        - 기존 머신러닝 방법에서는 이러한 특징 선택과 모델 학습의 과정이 나누어져 있지만, TabNet에서는 한 번에 가능
        - 특징 선택이 이루어지므로 어떠한 특징인지 설명이 가능함

## 12boostclass & 13boostclass : 캐글러에 일단 제출해보자

  > **Project**

## 14boostclass & 15boostclass : 경사하강법

  > **Project**

## 16boostclass : 딥러닝 학습방법 이해하기

   - **비선형모델, 신경망(neural network)**
     - **O** (n x p) **X** (n x d) **W**(d x p) = **b** **(n x p)**  : d개의 변수로 p개의 선형모델을 만들어서 p개의 잠재변수를 설명하는 모델을 상상해 볼 수 있음
     - **softmax 함수** : 모델의 출력을 확률로 해석할 수 있게 변환해주는 연산
      분류 문제를 풀 때 선형모델과 소프트맥스 함수를 결합해 예측함
          - softmax ( **o** ) = softmax (**Wx + b**)
          - 추론할 때는 one_hot 벡터로 최댓값을 가진 주소만 1로 출력하는 연산을 사용해서 softmax 사용하지 않음
      - 신경망은 **선형모델과 활성함수(activation function)를 합성한 함수**
          - 활성함수는 R 위에 정의된 **비선형함수**로서 딥러닝에서 매우 중요한 개념
          - **활성함수를 쓰지 않으면 딥러닝은 선형모형과 차이가 없음**
          - 시그모이드(sigmoid) 함수나 tanh함수는 전통적으로 많이 쓰이던 활성함수지만 **딥러닝에서는 RELU함수를 많이 쓰고 있음**
      - 다층(multi-layer) 퍼셉트론(MLP)은 **신경망이 여러층으로 합성된 함수**
          - 층이 깊을수록 **목적함수를 근사하는데 필요한 뉴런(노드)의 숫자가 훨씬 빨리 줄어들어 좀 더 효율적으로 학습이 가능함**
  - **딥러닝 학습원리 : 역전파 알고리즘**
      - 딥러닝은 **역전파(backpropagation) 알고리즘**을 이용하여 각 층에 사용된 패러미터를 학습함
      - 각 층 패러미터의 그레디언트 벡터는 윗층부터 역순으로 계산하게 됨
      - 역전파 알고리즘은 합성함수 미분법인 **연쇄법칙(chain-rule) 기반 자동 미분(auto-differentiation)을 사용함**
      
## 20boostclass & 21boostclass : 딥러닝 기초

- **딥러닝 기본 용어 설명**
   - 인공지능 : 사람의 지능을 모방 **>** 머신러닝  : data로 접근 **>** **딥러닝 : neural networks 구조 활용**
   -  **딥러닝의 주요 요소 4가지**
      - the **data** that the model can learn from
      - the **model** how to transform the data
      - the **loss function** that quantifies the badness of the model
      - the **algorithm** to adjust the parameters to minimize the loss

## 22boostclass & 23boostclass : 최적화

 - **최적화의 주요 용어 정리하기**
    - **Generalization**
       - Training error와 Test error의 차이
    - **Under- fitting vs over-fitting**
      - Over-fitting : 학습 데이터에 대해 잘 동작하지만 테스트 데이터에 대해서는 잘 동작하지 않는 현상
    - **Cross validation**
      - Cross-validation is a model validation technique for assessing how the model will generalize to an independent (test) data set
   - **Bias-Variance tradeoff**
   - **Bootstrapping**
     - Bootstrapping is any test or metric that uses random sampling with replacement
   - **Bagging and boosting**
    - Bagging ( Bootstrapping aggregating)
        - Multiple models are being trained with bootstrapping
        - ex) Base classifiers are fitted on random subset where individual predictions are aggregated(voting or averaging)
   - Boosting
        - It focuses on specific training samples that are hard to classify
        - A strong model is built by combining weak learners in sequence where each learner learns from the mistakes of the previous weak learner
 - **Gradient Descent Methods**
      - Stochastic gradient descent
          - Update with the gradient computed from a single sample
      - Mini-batch gradient descent
          - Update with the gradient computed from a subset of data
      - Batch gradient descent
          - Update with the gradient computed from the whole data
- **Regularization**
    - **Early stopping**
        - early stopping을 위해 additional validation data가 필요함
    - **Parameter norm penalty**
        - It adds smoothness to the function space
    - **Data augmentation**
        - More data are always welcomed
        - However, in most cases, training data are given in advance
        - In such cases, we need data augmentation
    - **Noise Robustness**
        - Add random noises inputs or weights
    - **Label Smoothing**
        - **Mix-up** constructs augmented training examples by mixing both input and output of two randomly selected training data
        - **CutMix** constructs augmented training examples by mixing inputs with cut paste and outputs with soft labels of two randomly selected training data
    - **Dropout**
        - In each forward pass, randomly set some neurons to zero
    - **Batch normalization**
        - Batch normalization compute the empirical mean and variance independently for each dimension (layers) and normalize
        - There are different variances of normalization
            
## 23boostclass & 24boostclass : 2-layer 인공신경망 구하기

  > project

## 25boostclass : CNN - Convloution은 무엇인가?

  - **Convolutional Neural Networks은 무엇인가?**
      - CNN consists of convolution layer, pooling layer, and fully connected layer
      - Convolution and pooling layers : feature extraction
      - Fully connected layer : decision making (e.g., classification)
          - Stride
          - Padding
      - 1 x 1 convolution
          - 이미지에서 한 픽셀만 보고 채널 방향으로 줄이는 것
          - why? dimension reduction
          - To reduce the number of parameters while increasing the depth
  - **Modern CNN - 1 x 1 convolution의 중요성**
      - depth는 깊어지는데 parameter 개수가 줄고 있는  테크닉에 집중
      - **ILSVRC**
          - ImageNet Large-Scale Visual Recognition Challenge
          - Classification / Detection / Localization / Segmenatation
          - 1,000 different categories
          - Over 1 million images
          - Training set : 456,567 images
          - **key ideas**
              - **RELU (Rectified Linear Unit) activation**
                  - Preserve properties of linear models
                  - Easy to optimize with gradient descent
                  - Overcome the vanishing gradient problem
              - GPU implementation (2 GPUs)
              - Local response normalization, Overlapping pooling
              - Data augmentation
              - Dropout
      - **VGGNet**
          - Increasing depth with **3 x 3 convolution** filters (with stride 1)
              - **why? Receptive field**
          - 1 x 1 convolution for fully connected layers
          - Dropout (p = 0.5)
          - VGG16, VGG19
      - **GoogleNet**
          - **Inception Block**
              - Reduce the number of parameter
              - How ? Recall how the number of parameters is computed
              **1 x 1 convolution** can be seen as channel-wise dimension reduction
      - **ResNet**
          - Deeper neural networks are hard to train
              - Overfitting is usually caused by an excessive number of parameters
          - Add an identity map **after** nonlinear activations
          - Batch normalization **after** convolutions
          - Bottleneck architecture
          - ***Performance*** increases while ***parameter*** size decrease
      - **DenseNet**
          - DenseNet uses **concatenation** instead of addition
          - **Dense Block > Transition Block 을 반복**

## 26boostclass : Computer Vision Aplications

- **Semantic Segmentation**
    - **Fully Convolution Network**
        - Transforming fully connected layers into convolution layers enables a classification net to output a heat map.
        - While FCN can run with inputs of any size, the output dimensions are typically reduced by subsampling.
        - So we need a way to connect the coarse output to the dense pixels.
    - **Deconvolution (conv transpose)**
- **Detection**
  - **R-CNN**
  - **SPPNet**
      - CNN runs once
  - **Faster R-CNN**
      - Region Proposal Network + Fast R-CNN
      - Region Proposal Network
          - 9 : Three different region sizes with three different ratios
          - 4 : four bounding box regression parameters
          - 2 : box classification
  - **YOLO**
    - YOLO(v1) is an extremely fast object detection algorithm
            - It simultaneously predict predicts multiple bounding boxes and class probabilities.

## 27boostclass : Sequential Model - RNN & Transformer

  - **Sequential Model**
      - **Autoregressive model**
          - fix the past timespan
      - **Markov model (first-order autoregressive model)**
          - 나의 현재는 과거에만 dependent !
          - Easy to express the joint distribution
      - **Latent autoregressive model**
          - summary of the past
  - **Recurrent Neural Network**
      - Short-term dependencies
      - **Long**-tern dependencies
  - **Long Short Term Memory**
      - core idea
          - **Forget Gate** : decide which information to throw away
          - **Input Gate** : decide with information to store in the cell state
  - **Gated Recurrent Unit**
      - Simpler architecture with two  gates(reset gate and update gate)
      - No cell state,  just hidden state

  - **Transformer**
    - Transformer is  the first sequence transduction model based entirely on attention
    - From a bird’s-eye view, this is what the Transformer does for machine translation tasks
    - If we glide down a little bit, this is what the Transformer does
    - The **Self-Attention** in both encoder and decoder is the cornerstone of Transformer.
        - First, we represent each word with embedding vectors
        - Then, Transformer encodes each word to feature vectors with **Self-Attention**
        - Suppose we are encoding two words : **Thinking** and **Machines**
        - Self-Attention at a high level
            - The animal didn’t cross the  street because **it** was too tired
        - Suppose we are encoding the first word : **Thinking** given ‘Thinking’ and ‘Machines’.
        - Then, we compute the **attention weights** by scaling followed by softmax.
        - Calculating Q, K, and V from X in a matrix form
        - Multi-headed attention (MHA) allows Transformer to focus on different positions
        - If eight heads are used, we end up getting eight different sets of encoded vectors (**attention heads**)
        - We simply pass them through additional (learnable) linear map
        - Why do we need positional encoding ?
            - n개의 단어를 sequential 하게 넣어줬다고 치지만 sequential한 정보가 이 안에 포함되어 있지 않음
            - This is the case for 512-dimensional encoding
            - Recent update on positional encoding
        - Now, let’s take a look at decoder side
        - Transformer transfers **key**(K) and **value**(V) of the topmost encoder to the decoder
        - The output sequence is generated in an autoregressive manner
        - In the **decoder**, the **self-attention layer** is only allowed to attend to earlier position in the output sequence which is done by masking future position before the softmax step
        - The “**Encoder-Decoder Attention**” layer works just like multi-headed self-attention, except it creates its Queries matrix from **the layer below it** and takes the **Keys** and **Values** form the encoder stack.
        - The final layer converts the stack of decoder outputs to the distribution over words
## 28boostclass : Generative Models

- **Generative Models 1**
    - We want to learn a probability distribution p(x) such that
        - **Generation** : If we sample x ~ p(x), x should look like a dog **(sampling)**
        - **Density estimation** : p(x) should be high if x looks like a dog, and low otherwise **(anomaly detection)**
            - Also known as, explicit models.
        - **Unsupervised representation learning** : We should be able to learn these images have in common **(feature learning)**
    - **Basic Discrete Distributions**
        - **Bernoulli distribution** : (biased) coin flip
        - **Categorical distribution** : (biased) m-side dice
    - **Conditional Independence**
        - **Chain rule**
            - p(x1) : 1 parameter
            - p(x2|x1) : 2 parameters ( one per p(x2|x1 = 0) and one per p(x2|x1 = 1))
            - p(x3|x1,x2) : 4 parameters
            - Hence, 1 + 2 + 2^2 + …. + 2^n -1 , which is the same as before
        - **Bayes’ rule**
        - **Conditional independence**
            - p(x1,….,xn) = p(x1)p(x2|x1)….p(xn|xn-1)
            - 2n - 1 parameters
            - Hence, by leveraging the Markov assumption, we get exponential reduction on the number of parameters
            - **Auto-regressive models** leverage this conditional independency
        - **Auto-regressive Model**
            - How can we parametrize **p(x)** ?
                - Let’s use the **chain rule** to factor the joint distribution
                - p(x1:784) = p(x1)p(x2|x1)p(x3|x1.2)..
                - This if called an **auto-regressive model**
                - Note that we need an ordering of all random variables
            - **NADE : Neural Autoregressive Density Estimator**
                - **NADE** is an **explicit** model that can compute the **density** of the given inputs
                - How can we compute the **density** of the given images?
                    - Suppose we have a binary image with 784 binary pixels
                    - Then, the joint probability is computed by
                        
                        p(x1,….x784) = p(x1)p(x2|x1)…p(x784|x1:783) 
                        
                - In case of modeling continuous random variables, **a mixture of Gaussian** can be used
            - **Pixel RNN**
                - We can also use **RNNs** to define an auto-regressive model
                - There are two model architectures in Pixel RNN based on the **ordering** of chain
                    - **Row LSTM**
                    - **Diagonal BiLSTM**
            
- **Generative Models 2**
    - **Variation Auto-encoder**
        - **Variational inference (VI)**
            - The goal if VI is to optimize **variational distribution** that best matches the **posterior distribution**
                - **Posterior distribution**
                - **Variational distribution**
            - In particular, we want to find the **variational distribution** that minimizes the KL divergence between the true posterior
        - **Key limitation**
            - It is an **intractable** model(hard to evaluate likelihood)
            - The prior fitting term must be differentiable, hence it is hard to use diverse latent prior distributions
            - In most cases, we use an isotropic Gaussian
        - **Adversarial Auto-encoder**
            - It allows us to use arbitrary latent distributions that we can sample
    - **GAN**
        - A two player minimax game between **generator** and **discriminator**
  
## 29boostclass : 딥러닝 모델 구현하기

  > project
