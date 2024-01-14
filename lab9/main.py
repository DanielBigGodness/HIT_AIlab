# 1. 装载英文bert tokenizer 和 bert-base-cased模型
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

# 2. 用Bert为句子"Apple company does not sell the apple."编码
sentence = "Apple company does not sell the apple."
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)

# 3. 输出句子转化后的ID
print(inputs['input_ids'])
# tensor([[ 101, 7277, 1419, 1185, 1256, 1176, 2356, 1103, 7277, 119,  102]])

# 4. 分别输出句子编码后单词‘CLS’,‘Apple’,‘apple’和‘SEP’,四个词对应的编码
encoded_cls = outputs.last_hidden_state[:,0,:]
encoded_apple1 = outputs.last_hidden_state[:,1,:]
encoded_apple2 = outputs.last_hidden_state[:,-2,:]
encoded_sep = outputs.last_hidden_state[:,-1,:]
print(encoded_cls)
print(encoded_apple1)
print(encoded_apple2)
print(encoded_sep)

# 5. 分别计算‘Apple’和‘apple’，‘CLS’和‘Apple’,‘CLS’和‘SEP’之间的距离
from torch.nn.functional import pairwise_distance
distance_apple = pairwise_distance(encoded_apple1, encoded_apple2)
distance_cls_apple = pairwise_distance(encoded_cls, encoded_apple1)
distance_cls_sep = pairwise_distance(encoded_cls, encoded_sep)
print(distance_apple)
print(distance_cls_apple)
print(distance_cls_sep)

# 6. 输入句子“I have a [MASK] named Charlie.”，重新加载BertForMaskedLM模型，通过bert 预测[mask] 位置最可能的单词
masked_sentence = "I have a [MASK] named Charlie."
masked_inputs = tokenizer(masked_sentence, return_tensors='pt')
masked_model = BertForMaskedLM.from_pretrained('bert-base-cased')
masked_outputs = masked_model(**masked_inputs).logits
predicted_index = torch.argmax(masked_outputs[0, masked_inputs['input_ids'][0] == tokenizer.mask_token_id])
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
# dog


#7
masked_sentence = "I have a cat [MASK]"
for i in range(10):
  masked_inputs = tokenizer(masked_sentence, return_tensors='pt')
  masked_model = BertForMaskedLM.from_pretrained('bert-base-cased')
  masked_outputs = masked_model(**masked_inputs).logits
  predicted_index = torch.argmax(masked_outputs[0, masked_inputs['input_ids'][0] == tokenizer.mask_token_id])
  predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
  if predicted_token == "[PAD]":
    break
  masked_sentence = masked_sentence.strip('[MASK].')
  masked_sentence = masked_sentence.strip('[MASK] .')
  masked_sentence = masked_sentence.strip('[MASK]')
  # masked_sentence += " "
  masked_sentence += predicted_token
  print(masked_sentence)
  # masked_sentence += " "
  masked_sentence += '[MASK]'
  # masked_sentence += " "

print(masked_sentence)

