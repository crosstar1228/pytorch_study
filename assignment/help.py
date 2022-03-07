# !pip install transformers
# contain 2 function : set_device, custom_collate_fn 
# contain 2 class : customdataset, customclassifier
# device setting
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import BertTokenizer, BertModel

tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    return device

# 전처리 함수
def custom_collate_fn(batch):

  """
  - batch: list of tuples (input_data(string), target_data(int))
  
  한 배치 내 문장들을 tokenizing 한 후 텐서로 변환함. 
  이때, dynamic padding (즉, 같은 배치 내 토큰의 개수가 동일할 수 있도록, 부족한 문장에 [PAD] 토큰을 추가하는 작업)을 적용
  토큰 개수는 배치 내 가장 긴 문장으로 해야함.
  또한 최대 길이를 넘는 문장은 최대 길이 이후의 토큰을 제거하도록 해야 함
  토크나이즈된 결과 값은 텐서 형태로 반환하도록 해야 함
  
  한 배치 내 레이블(target)은 텐서화 함.
  
  (input, target) 튜플 형태를 반환.
  """
  global tokenizer_bert
  
  input_list, target_list = zip(*batch)
  
  tensorized_input = tokenizer_bert(
      text=input_list,
      padding='longest',
      add_special_tokens=True,
      truncation=True, #longest는 최대길이를 초과함
      return_tensors='pt'
  )
  
  tensorized_label = torch.tensor(target_list)
  
  return tensorized_input, tensorized_label


#  dataset
class CustomDataset(Dataset):
  """
  - input_data: list of string
  - target_data: list of int
  """
  def __init__(self, input_data:list, target_data:list) -> tuple:
      self.X = input_data
      self.Y = target_data

  def __len__(self):
      return len(self.Y)

  def __getitem__(self, index):
      return self.X[index], self.Y[index]

## classifier
class CustomClassifier(nn.Module):

  def __init__(self, hidden_size: int, n_label: int):
    super(CustomClassifier, self).__init__()

    self.bert = BertModel.from_pretrained("klue/bert-base")

    dropout_rate = 0.1
    linear_layer_hidden_size = 32

    # torch.nn에서 제공되는 Sequential, Linear, ReLU, Dropout 함수 활용
    self.classifier = nn.Sequential(
          nn.Linear(hidden_size, linear_layer_hidden_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(linear_layer_hidden_size,n_label)
        )

  def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    # BERT 모델의 마지막 레이어의 첫번재 토큰을 인덱싱
    cls_token_last_hidden_states = outputs['pooler_output'] # 마지막 layer의 첫 번째 토큰 ("[CLS]") 벡터를 가져오기, shape = (1, hidden_size)
    # print(cls_token_last_hidden_states.shape)
    logits = self.classifier(cls_token_last_hidden_states)
    # print(cls_token_last_hidden_states)
    return logits