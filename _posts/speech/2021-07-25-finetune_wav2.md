---
layout: post
comments: true
title: Fairseq 코드리뷰 Wav2vec 2.0 (Finetune)
categories: Fairseq

tags:
- Speech Recognition
---

**<span style='color:DarkRed'>Fairseq의 Wav2vec 2.0 Finetune 실행방법 </span>**

> Fairseq의 제공하는 Wav2vec 2.0 모델의 작동과정을 소개하고자 합니다.
- 본 게시글은 아래의 github를 참고하였습니다.
    - https://github.com/pytorch/fairseq.git
    - https://github.com/mailong25/self-supervised-speech-recognition.git

**<span style='color:DarkRed'>Finetuning using self-supervised learning </span>**

- `audio file path`과 대응되는 `trascription`을 준비

```
/examples/label_data/audios/LJ001-0002.wav	IN BEING COMPARATIVELY MODERN
/examples/label_data/audios/LJ001-0013.wav	THAN IN THE SAME OPERATIONS WITH UGLY ONES
/examples/label_data/audios/LJ001-0025.wav	IMITATES A MUCH FREER HAND SIMPLER ROUNDER AND LESS SPIKY AND THEREFORE FAR PLEASANTER AND EASIER TO READ
/examples/label_data/audios/LJ001-0030.wav	A VERY FEW YEARS SAW THE BIRTH OF ROMAN CHARACTER NOT ONLY IN ITALY BUT IN GERMANY AND FRANCE
/examples/label_data/audios/LJ001-0041.wav	IT MUST BE SAID THAT IT IS IN NO WAY LIKE THE TRANSITION TYPE OF SUBIACO
/examples/label_data/audios/LJ001-0042.wav	AND THOUGH MORE ROMAN THAN THAT YET SCARCELY MORE LIKE THE COMPLETE ROMAN TYPE OF THE EARLIEST PRINTERS OF ROME
/examples/label_data/audios/LJ001-0048.wav	HIS LETTER IS ADMIRABLY CLEAR AND REGULAR BUT AT LEAST AS BEAUTIFUL AS ANY OTHER ROMAN TYPE
/examples/label_data/audios/LJ001-0051.wav	AND PAYING GREAT ATTENTION TO THE PRESS WORK OR ACTUAL PROCESS OF PRINTING
/examples/label_data/audios/LJ001-0064.wav	MANY OF WHOSE TYPES INDEED LIKE THAT OF THE SUBIACO WORKS ARE OF A TRANSITIONAL CHARACTER
/examples/label_data/audios/LJ001-0086.wav	ARE DAZZLING AND UNPLEASANT TO THE EYE OWING TO THE CLUMSY THICKENING AND VULGAR THINNING OF THE LINES
```

- 주어진 text에 대한 dictionary를 미리 만들어야 됨
    - `finetune_dir/dict.ltr.txt`에 character단위 글자을 저장

```python
words = [d.split('\t')[1].upper() for d in data]
'''
words
['IN BEING COMPARATIVELY MODERN', 'THAN IN THE SAME OPE... UGLY ONES', 'IMITATES A MUCH FREE...ER TO READ', 'A VERY FEW YEARS SAW...AND FRANCE', 'IT MUST BE SAID THAT...OF SUBIACO', 'AND THOUGH MORE ROMA...RS OF ROME', 'HIS LETTER IS ADMIRA...ROMAN TYPE', 'AND PAYING GREAT ATT...F PRINTING', 'MANY OF WHOSE TYPES ... CHARACTER', 'ARE DAZZLING AND UNP... THE LINES']
'''
letters = [d.replace(' ','|') for d in words]
'''
letters
['IN|BEING|COMPARATIVELY|MODERN', 'THAN|IN|THE|SAME|OPE...|UGLY|ONES', 'IMITATES|A|MUCH|FREE...ER|TO|READ', 'A|VERY|FEW|YEARS|SAW...AND|FRANCE', 'IT|MUST|BE|SAID|THAT...OF|SUBIACO', 'AND|THOUGH|MORE|ROMA...RS|OF|ROME', 'HIS|LETTER|IS|ADMIRA...ROMAN|TYPE', 'AND|PAYING|GREAT|ATT...F|PRINTING', 'MANY|OF|WHOSE|TYPES|...|CHARACTER', 'ARE|DAZZLING|AND|UNP...|THE|LINES']
'''
letters = [' '.join(list(d)) + ' |' for d in letters]
'''
letters
['I N | B E I N G | C ... D E R N |', 'T H A N | I N | T H ... O N E S |', 'I M I T A T E S | A ... R E A D |', 'A | V E R Y | F E W ... A N C E |', 'I T | M U S T | B E ... I A C O |', 'A N D | T H O U G H ... R O M E |', 'H I S | L E T T E R ... T Y P E |', 'A N D | P A Y I N G ... T I N G |', 'M A N Y | O F | W H ... C T E R |', 'A R E | D A Z Z L I ... I N E S |']
'''
chars = [l.split() for l in letters]
'''
chars
[['I', 'N', '|', 'B', 'E', 'I', 'N', 'G', '|', ...], ['T', 'H', 'A', 'N', '|', 'I', 'N', '|', 'T', ...], ['I', 'M', 'I', 'T', 'A', 'T', 'E', 'S', '|', ...], ['A', '|', 'V', 'E', 'R', 'Y', '|', 'F', 'E', ...], ['I', 'T', '|', 'M', 'U', 'S', 'T', '|', 'B', ...], ['A', 'N', 'D', '|', 'T', 'H', 'O', 'U', 'G', ...], ['H', 'I', 'S', '|', 'L', 'E', 'T', 'T', 'E', ...], ['A', 'N', 'D', '|', 'P', 'A', 'Y', 'I', 'N', ...], ['M', 'A', 'N', 'Y', '|', 'O', 'F', '|', 'W', ...], ['A', 'R', 'E', '|', 'D', 'A', 'Z', 'Z', 'L', ...]]
'''
chars = [j for i in chars for j in i]
char_stats = list(Counter(chars).items())
char_stats = sorted(char_stats, key=lambda x : x[1], reverse = True)
'''
char_stats
[('|', 149), ('E', 76), ('A', 72), ('T', 67), ('N', 55), ('R', 52), ('I', 47), ('O', 44), ('S', 37), ('H', 29), ('L', 26), ('Y', 21), ('M', 19), ('D', 18), ...]
'''
char_stats = [c[0] + ' ' + str(c[1]) for c in char_stats]
'''
char_stats
['| 149', 'E 76', 'A 72', 'T 67', 'N 55', 'R 52', 'I 47', 'O 44', 'S 37', 'H 29', 'L 26', 'Y 21', 'M 19', 'D 18', ...]
'''
```

- 학습시킬 데이터를 `train`, `valid`의 형태로 나눔

```bash
python3 gen_dict.py --transcript_file path/to/transcript.txt --save_dir path/to/save_dir
```

```bash
manifest/
├── dict.ltr.txt
├── train.ltr
├── train.tsv
├── train.wrd
├── valid.ltr
├── valid.tsv
└── valid.wrd
```

```python
# train.ltr
M A N Y | O F | W H O S E | T Y P E S | I N D E E D | L I K E | T H A T | O F | T H E | S U B I A C O | W O R K S | A R E | O F | A | T R A N S I T I O N A L | C H A R A C T E R |
T H A N | I N | T H E | S A M E | O P E R A T I O N S | W I T H | U G L Y | O N E S |
...
# train.tsv
/home/donghwa/Documents/PR/self-supervised-speech-recognition/examples/label_data/audios/LJ001-0064.wav	94109
/home/donghwa/Documents/PR/self-supervised-speech-recognition/examples/label_data/audios/LJ001-0013.wav	41353
...
# train.wrd
MANY OF WHOSE TYPES INDEED LIKE THAT OF THE SUBIACO WORKS ARE OF A TRANSITIONAL CHARACTER
THAN IN THE SAME OPERATIONS WITH UGLY ONES
...
```
- 학습데이터의 시간이 1시간인지, 2시간인지 판단
    - 아래의 예시는 1시간이내의 경우로 `base_1h.yaml`를 사용하여 finetuning

```yaml
# @package _group_

common:
  fp16: true
  log_format: json
  log_interval: 200

checkpoint:
  save_interval: 1000
  save_interval_updates: 50
  keep_interval_updates: 1
  no_epoch_checkpoints: true
  best_checkpoint_metric: wer

task:
  _name: audio_pretraining
  data: ???
  normalize: false
  labels: ltr
  sample_rate: 16000
  autoregressive: false

dataset:
  num_workers: 6
  max_tokens: 2800000
  skip_invalid_size_inputs_valid_test: true
  validate_after_updates: 1000
  validate_interval: 1000
  valid_subset: valid

distributed_training:
  ddp_backend: no_c10d
  distributed_world_size: 2

criterion:
  _name: ctc
  zero_infinity: true

optimization:
  max_update: 13000
  lr: [0.00005]
  sentence_avg: true
  update_freq: [4]

optimizer:
  _name: adam
  adam_betas: (0.9,0.98)
  adam_eps: 1e-08

lr_scheduler:
  _name: tri_stage
  phase_ratio: [0.1, 0.4, 0.5]
  final_lr_scale: 0.05

model:
  _name: wav2vec_ctc
  w2v_path: ???
  apply_mask: true
  mask_prob: 0.65
  mask_channel_prob: 0.25
  mask_channel_length: 64
  layerdrop: 0.1
  activation_dropout: 0.1
  feature_grad_mult: 0.0
  freeze_finetune_updates: 1000
```

**<span style='color:DarkRed'> Finetuning procedure  </span>**


- 데이터 불러오기 from `FileAudioDataset`

```python
def __getitem__(self, index):
    import soundfile as sf

    fname = os.path.join(self.root_dir, self.fnames[index])
    wav, curr_sample_rate = sf.read(fname)
    feats = torch.from_numpy(wav).float()
    feats = self.postprocess(feats, curr_sample_rate) # 정규화
    return {"id": index, "source": feats}
```

- 타켓 데이터 추가 from `AddTargetDataset`
- `item` 위에서 생성된 `id`와 오디오 `source`

```python
item = self.dataset[index]
item["label"] = self.get_label(index)

# in self.get_label
self.process_label(self.labels[index])

# in self.process_label
self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )
```
- line 별로 자소단위로 분리된 텍스트를 불러옴

```python

# line
'I M I T A T E S | A | M U C H | F R E E R | H A N D | S I M P L E R | R O U N D E R | A N D | L E S S | S P I K Y | A N D | T H E R E F O R E | F A R | P L E A S A N T E R | A N D | E A S I E R | T O | R E A D |\n'

# in encode_line
words = line_tokenizer(line)

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

# words
['I', 'M', 'I', 'T', 'A', 'T', 'E', 'S', '|', 'A', '|', 'M', 'U', 'C', ...]
```

- Token indexing
    - specical token 사전에 4개가 추가됨 

```python
'''self.indices
{'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'A': 6, 'T': 7, 'N': 8, 'R': 9, 'I': 10, 'O': 11, 'S': 12, 'H': 13, ...}
'''
if sym in self.indices:
    return self.indices[sym]
```

```python
for i, word in enumerate(words):
    if add_if_not_exist:
        idx = self.add_symbol(word)
    else:
        idx = self.index(word)

ids
tensor([10, 16, 10,  7,  6,  7,  5, 12,  4,  6,  4, 16, 20, 18, 13,  4, 21,  9,
         5,  5,  9,  4, 13,  6,  8, 17,  4, 12, 10, 16, 19, 14,  5,  9,  4,  9,
        11, 20,  8, 17,  5,  9,  4,  6,  8, 17,  4, 14,  5, 12, 12,  4, 12, 19,
        10, 25, 15,  4,  6,  8, 17,  4,  7, 13,  5,  9,  5, 21, 11,  9,  5,  4,
        21,  6,  9,  4, 19, 14,  5,  6, 12,  6,  8,  7,  5,  9,  4,  6,  8, 17,
         4,  5,  6, 12, 10,  5,  9,  4,  7, 11,  4,  9,  5,  6, 17,  4],
```

- 생성된 token index를 item에 붙여 주게 됨

```python
item["label"] = self.get_label(index)

item
{'id': 5, 'source': tensor([-2.0142e-03,...0000e+00]), 'label': tensor([10, 16, 10, ...rch.int32)}
```

- conv layer 적용, no_grad로 학습시키지 않음

```python
'''
source
tensor([[-2.1362e-04,  0.0000e+00, -9.1553e-05,  ..., -9.7656e-04,
         -1.0071e-03,  0.0000e+00]], device='cuda:0', dtype=torch.float16)
source.shape
torch.Size([1, 30393])
'''
with torch.no_grad():
    features = self.feature_extractor(source)
'''
features.shape
torch.Size([1, 94, 768])
'''
```
- transformer 적용

```python
'''
input x shape: torch.Size([1, 94, 768])
'''
x = self.extract_features(x, padding_mask)
'''
output x shape: torch.Size([1, 94, 768])
'''
```

- finetuning 단계에서 새롭게 추가되는 부분

```python
'''
x.shape 
torch.Size([1, 94, 768]) # embedding from the transformer
padding_mask.shape  # frame mask which occur
torch.Size([1, 94])
''''

x = x.transpose(0, 1)
'''
x.shape 
torch.Size([94, 1, 768]) => (seq, bz, dim)
'''
```


- linear projection
    - character vocab size에 따라 final dim이 결정됨

```python
if tgt_dict is not None:
    self.proj = Linear(d, len(tgt_dict))

x.shape 
torch.Size([94, 1, 28])
```


- 타켓확률 계산

```python
lprobs = model.get_normalized_probs(
    net_output, log_probs=True
).contiguous()  # (T, B, C) from the encoder

def get_normalized_probs(self, net_output, log_probs):
    """Get normalized probabilities (or log probs) from a net's output."""

    logits = net_output["encoder_out"]
    if log_probs:
        return utils.log_softmax(logits.float(), dim=-1)
'''
utils.log_softmax(logits.float(), dim=-1).shape 
torch.Size([94, 1, 28])
'''
```



- 타켓 전처리 & CTC loss 계산
    - task.blank_symbol = `<s>` 로 0번째 index로 설정이 되어 있음 

```python
'''
# targets_flat
tensor([10,  8,  4, 23,  5, 10,  8, 22,  4, 18, 11, 16, 19,  6,  9,  6,  7, 10,
        26,  5, 14, 15,  4, 16, 11, 17,  5,  9,  8,  4], device='cuda:0',
       dtype=torch.int32)

# target_lengths
tensor([30], device='cuda:0')
'''

with torch.backends.cudnn.flags(enabled=False):
    loss = F.ctc_loss(
        lprobs, # torch.Size([94, 1, 28])
        targets_flat, # torch.Size([30])
        input_lengths, # 94
        target_lengths, # 30
        blank=self.blank_idx, # self.blank_idx: 0
        reduction="sum",
        zero_infinity=self.zero_infinity,
    )
```

- CTC에 대한 자세한 설명은 PDF [link](https://drive.google.com/file/d/1pMZ3zS9DLXborJjTnw0xxjvT1uO93vbM/view?usp=sharing) 참조