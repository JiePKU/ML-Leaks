
### An unoffcial implementation for "ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models"

### Requirement

python=3.8.13

torch=1.12.0

torchvision=0.13.0

numpy=1.22.3

scikit-learn=1.2.0


### Quick Start

```python
python infer.py --dataset cifar10  --data_path '/path/to/your/data/' --num_epoch 50
```

#### Result

```
Shadow model training
Testing Accuracy: 0.6127
More detailed results:
              precision    recall  f1-score   support

           0       0.68      0.66      0.67      1263
           1       0.71      0.74      0.73      1216
           2       0.50      0.46      0.48      1257
           3       0.42      0.44      0.43      1219
           4       0.56      0.52      0.54      1294
           5       0.51      0.49      0.50      1256
           6       0.67      0.74      0.71      1261
           7       0.67      0.64      0.65      1218
           8       0.71      0.76      0.74      1258
           9       0.68      0.68      0.68      1258

    accuracy                           0.61     12500
   macro avg       0.61      0.61      0.61     12500
weighted avg       0.61      0.61      0.61     12500

Target model training
Testing Accuracy: 0.6105
More detailed results:
              precision    recall  f1-score   support

           0       0.67      0.64      0.66      1259
           1       0.68      0.75      0.71      1257
           2       0.47      0.48      0.47      1216
           3       0.44      0.42      0.43      1247
           4       0.54      0.48      0.51      1202
           5       0.52      0.53      0.53      1261
           6       0.66      0.72      0.69      1275
           7       0.70      0.66      0.68      1265
           8       0.75      0.74      0.74      1278
           9       0.66      0.68      0.67      1240

    accuracy                           0.61     12500
   macro avg       0.61      0.61      0.61     12500
weighted avg       0.61      0.61      0.61     12500

Attacker of adversary 1 training
Testing Accuracy: 0.6939
More detailed results:
              precision    recall  f1-score   support

           0       0.81      0.50      0.62     12500
           1       0.64      0.88      0.74     12500

    accuracy                           0.69     25000
   macro avg       0.73      0.69      0.68     25000
weighted avg       0.73      0.69      0.68     25000

```
