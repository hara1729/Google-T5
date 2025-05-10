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

--------------------------------------
Example Training Logs:
--------------------------------------

Epoch 1/20

318/318 ━━━━━━━━━━━━━━━━━━━━ 231s 605ms/step - loss: 7.4395 - val_accuracy: 0.1344 - val_loss: 6.7044
Epoch 2/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 180s 567ms/step - loss: 6.4076 - val_accuracy: 0.1695 - val_loss: 6.3738
Epoch 3/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 182s 572ms/step - loss: 6.0720 - val_accuracy: 0.1974 - val_loss: 6.1071
Epoch 4/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 179s 561ms/step - loss: 5.8030 - val_accuracy: 0.2123 - val_loss: 5.9241
Epoch 5/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 180s 565ms/step - loss: 5.5956 - val_accuracy: 0.2278 - val_loss: 5.7946
Epoch 6/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 183s 574ms/step - loss: 5.4205 - val_accuracy: 0.2322 - val_loss: 5.6956
Epoch 7/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 188s 591ms/step - loss: 5.2674 - val_accuracy: 0.2409 - val_loss: 5.6035
Epoch 8/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 187s 587ms/step - loss: 5.1277 - val_accuracy: 0.2455 - val_loss: 5.5299
Epoch 9/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 185s 580ms/step - loss: 4.9939 - val_accuracy: 0.2495 - val_loss: 5.4813
Epoch 10/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 186s 584ms/step - loss: 4.8643 - val_accuracy: 0.2577 - val_loss: 5.4179
Epoch 11/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 187s 588ms/step - loss: 4.7483 - val_accuracy: 0.2587 - val_loss: 5.3658
Epoch 12/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 187s 587ms/step - loss: 4.6392 - val_accuracy: 0.2617 - val_loss: 5.3646
Epoch 13/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 185s 583ms/step - loss: 4.5309 - val_accuracy: 0.2624 - val_loss: 5.3847
Epoch 14/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 185s 582ms/step - loss: 4.4257 - val_accuracy: 0.2693 - val_loss: 5.3471
Epoch 15/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 186s 585ms/step - loss: 4.3255 - val_accuracy: 0.2726 - val_loss: 5.3195
Epoch 16/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 189s 593ms/step - loss: 4.2245 - val_accuracy: 0.2705 - val_loss: 5.3163
Epoch 17/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 191s 601ms/step - loss: 4.1308 - val_accuracy: 0.2711 - val_loss: 5.3159
Epoch 18/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 194s 610ms/step - loss: 4.0406 - val_accuracy: 0.2730 - val_loss: 5.3622
Epoch 19/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 193s 605ms/step - loss: 3.9481 - val_accuracy: 0.2777 - val_loss: 5.3658
Epoch 20/20
318/318 ━━━━━━━━━━━━━━━━━━━━ 186s 584ms/step - loss: 3.8634 - val_accuracy: 0.2804 - val_loss: 5.3487
