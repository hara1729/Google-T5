This is a very simple implemetation of Google's T5 Model from scratch using `TensorFlow/keras` on the `WikiQA` dataset using context fetching via RAG.

--------------------------------------
To Run
--------------------------------------
  1. `pip install -r requirements.txt`
  2. python nn.py

--------------------------------------
Note:
--------------------------------------
  - Almost all hyper-parameters of the T5 model are exposed via `T5Model()` . You may choose to expeiment with any hyper-parameter combination of your will.
  - Additionally this project implements `GroupedQueryAttenion` which the T5 model predates. You can set `attention_type = 'gqa'` to use grouped query attention.
  - Don't expect this model to hit benchmark accuracies since the dataset is very small and the model size is huge (even for the T5-mini model) compared to the data volume.         Usually these kinds of model require a lot of "pre-training" to perform well. You should expect the accuracies to saturate at less that 30% accuracy.
  - This project gives an idea about RAG and custom coding LLMs from scratch. Nothing Else!!!
