from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
while(True):
  example = input ("Input:") # "My name is Wolfgang and I live in Berlin"
  if exexample=="quit":
    break
  ner_results = nlp(example)
  print(ner_results)
