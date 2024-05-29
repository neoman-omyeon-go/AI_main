# AI_main

# Object Detection
AI는 영양 정보를 얻는 기능에서 주요 기능을 수행한다. 사용자가 영양 정보의 사진을 촬영하거나 갤러리에 있는 사진에 있는 영양 정보들을 얻기 위해서 AI가 사용된다. 사용자에 의해서 찍어지는 사진을 통해서 영양 정보 테이블의 위치를 추출한다. 영양 정보 테이블 추출은 MS-COCO 데이터셋으로 학습된 Ultralytics 의 YOLOv5 모델이 아닌 우리의 Task에 맞게 테이블을 추출하기 위해서 직접 영양 정보 테이블을 가지는 데이터들을 수집하고, 데이터로부터 테이블을 탐지하기 위한 라벨링을 직접 수행하고 라벨링을 통해서 학습한 YOLOv5 모델을 사용하여 영양 정보 테이블을 식별하고 위치를 추출한다. 

![image](https://github.com/neoman-omyeon-go/AI_main/assets/97443033/7f2c554f-bd85-49d8-a18f-a10cf593dd58)
![image](https://github.com/neoman-omyeon-go/AI_main/assets/97443033/254c3d9a-4e6e-4d39-9f5b-0b7d5e13ee48)

YOLO모델은 Ultralytics에서 제공하는 오픈 소스를 이용하여 개발하였다. 모델 구성은 다음과 같다.
	Backbone Network(이미지 Feature 추출) : CSPDarknet53 (CNN + Cross Stage Partial Network)
	Neck(추출된 특징들을 조합하고 확장) : PANet (Path Aggregation Network)
	Head(객체의 클래스, 위치, 크기 예측)

# CRAFT
영양 정보 테이블이 정확히 식별되고 난 후, CRAFT 기술을 통해 사진 속에서 텍스트의 정확한 위치를 파악한다. 이후 잘려지는 이미지의 부분과 좌표값을 얻게된다. 이 과정에서도 영양정보 테이블에서 어떻게 하면 TEXT 잘 잘리는 지에 대해서 직접 실험을 돌려 하이퍼 파라미터를 선정하였다. CRAFT된 이미지들의 좌표의 값을 기준으로 우리가 만들어낸 정렬하는 알고리즘을 사용하여 텍스트가 무작위로 잘려서 저장되는 게 아닌 왼쪽위에서부터 오른쪽으로 사람이 읽는 순서대로 이미지를 저장하였다. 

CFART모델은 CLOVA에서 제공하는 오픈 소스를 이용하여 개발하였다. 모델 구성은 다음과 같다.
	Backbone Network : VGG16 (여러 컨볼루션 레이어를 통해 이미지에서 고수준의 특징을 추출한다. Feature map : 이미지의 다양한 부분에서 텍스트가 존재할 가능성에 대한 정보를 담는다.)
	Feature Linking : 각 문자의 영역을 인식하는 것뿐만 아니라 문자가 어떻게 연결되어 단어나 문장을 형성하는지를 탐지한다.
	Double Convolution Module : 특징 추출 후, 모델은 더블 컨볼루션 모듈을 사용하여 Feature map을 더 정제한다. 추가적으로 이 모듈은 텍스트의 경계를 더 세밀하게 파악하는데 도움을 준다.

# OCR
OCR 에서 들어갈 때 출력 값을 txt 파일로 저장할 것이다. 이 정보에 순서에 따라 예를들어 ‘나트륨’이라는 정보다음에 나오는 수치가 기입이 되는데 순서가 바뀌면 입력에 오류가 생긴다. 그렇기 때문에 출력결과에서 큰 영향을 미친다.
이렇게 나온 출력 값을 후처리 기술을 통해 단순한 ‘륨’을 ‘룸’으로 예측하는 오타들은 처리할 수 있게 만들어 적용하여 최종적으로 dictionary 형태로 저장하였다.

OCR 모델은 CLOVA에서 제공하는 오픈 소스를 이용하여 개발하였다. 모델 구성은 다음과 같다
	Transformation: TPS
	이미지 Feature 추출: ResNet
	Sequence Modeling: LSTM
	Prediction: Attn

여러 조합을 테스트한 결과, 이 구성으로 최적의 성능을 얻었다. 또한, feature 추출에 RCNN과 VGG를 사용해보고, prediction에 CTC를 사용해보았으나, 최종적으로 선택한 조합이 가장 우수한 성능을 보였다.
모델 학습을 위해 데이터를 직접 구축하였습니다. 데이터는 2023 교원 그룹 AI OCR에서 제공하는 한글 데이터 76,887장과 직접 생성한 영어 이미지 30,000장, 숫자 이미지 30,000장, 영양 테이블 표에 있는 텍스트 10,000장으로 구성된 데이터셋을 사용하였습니다. 초기에는 한글 가공식품 데이터만 학습하였으나, 성능이 좋지 않았다. 이는 수치 뒤에 mg, g, %, 등 영어가 섞여 있기 때문임을 확인하였다. 이후 숫자와 영어 데이터를 추가로 생성하여 학습시켰지만, 성능 향상은 미미했다.
마지막으로, 실제 사용할 영양성분표를 가져와 10,000장을 라벨링하여 학습한 결과, 성능이 크게 향상되었다.

![image](https://github.com/neoman-omyeon-go/AI_main/assets/97443033/33835cc3-15c8-4bdc-bc3a-85098ea1dadd)
![image](https://github.com/neoman-omyeon-go/AI_main/assets/97443033/be4bc208-52a1-4dd1-b9a0-d7f5c5f51245)

아래는 우리 AI 모델의 전체적인 구조이다. 사용자가 사진을 찍으면, 먼저 객체 감지(Object Detection)를 사용하여 이미지를 크롭한다. 크롭된 이미지를 이용해 텍스트 감지(Text Detection)를 수행한 후, 정렬 알고리즘을 통해 감지된 텍스트를 정렬한다. 마지막으로, OCR을 통해 텍스트를 인식하고, 탄수화물, 단백질, 당류, 지방, kcal, 콜레스테롤, 나트륨 정보를 딕셔너리 형태로 저장한다.

![image](https://github.com/neoman-omyeon-go/AI_main/assets/97443033/49a7ca8d-87de-4981-8aa5-57e055cb8a21)



