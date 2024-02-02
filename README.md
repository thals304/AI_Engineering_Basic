# AI_Engineering_Basic
**[naver connect] boostcourse - AI 엔지니어 기초 다지기**

<img src=".png" width="50" height="50"/>

## ⏱️스터디 기간

2024.01.05 ~ 2024.02.27

## ⚙️개발 환경

- anaconda3 1.12.1

- jupyter notebook 6.5.4 (Phython 3.12.1)

## 01boostclass : numpy & pandas

- **numpy**

- **pandas**

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

  
