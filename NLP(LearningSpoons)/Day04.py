# tf-idf
# tf(t,D) : term-doc frequency 혹은 나오면 1안나오면0도 가능, 횟수의 로그 스케일도 사용
# df(t,D) : doc freq, 단어가 포함된 문서 빈도

# Text similarity
"""
text similarity demo : https://dandelion.eu/semantic-text/text-similarity-demo/?text1=Reports+that+the+NSA+eavesdropped+on+world+leaders+have+%22severely+shaken%22+relations+between+Europe+and+the+U.S.%2C+German+Chancellor+Angela+Merkel+said.&text2=Germany+and+France+are+to+seek+talks+with+the+US+to+settle+a+row+over+spying%2C+as+espionage+claims+continue+to+overshadow+an+EU+summit+in+Brussels.&lang=auto&exec=true
텍스트 유사도 데이터셋
SentEval : sentence embedding을 ㅍ평가하기 위한 데이터셋

jaccard sim vs distance
distance : cosine, 유클리디언, 맨하탄
jaccard : 교집합

cosine distance는 거리의 4가지 기준을 만족하지 않는다. 그래서 원래 거리는 아닌데 그렇게 부름. 엄밀히는 그냥 유사도

MaLSTM : 맨하탄 거리를 사용

Natural Language Inference
: neutral, entail, contradiction의 분류를 통해 언어 임베딩?


"""