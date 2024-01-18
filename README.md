# AI_Engineering_Basic
**[naver connect] boostcourse - AI 엔지니어 기초 다지기**

## ⏱️스터디 기간

2024.01.05 ~ 2024.02.27

## ⚙️개발 환경

- anaconda3 1.12.1

- jupyter notebook 6.5.4 (Phython 3.12.1)

## 01boostclass : numpy & pandas

**numpy**

**pandas**

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

## 03boostclass : EDA



