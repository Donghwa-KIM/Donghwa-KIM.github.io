ppl perplexity 
gN 가우시안 노이즈
bleu score
163.152.184.97

- Tokenize
I am a student
- [‘I’, ‘am’, ‘a’, ‘student’]
- [‘yes’,‘i’, ‘am’  ]

‘Pad token’ 짧은 문장
먼거리에 있는 몇개의 단어는 불필요 긴문장

i
Bucketing
Padding
Dynamic RNN 10~20 words
Bi-directional RNN, 앞에 있는 정보가 중요한 경우/ 뒤에 정보가 중요한 경우 


Bleu score
영어 to 프랑스 : 한 두개의 적절한 번역
영어 to 영어: comment / response  동시발생도 없고,  bleu score에 커지길바래는데 낮은 값들을 가지면 오버핏일 경우가 있음
Output.dev: 


# inference
Bad response along score
Remove punctuation


