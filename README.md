# readability


* Single layer of BERT
```python
python train.py --lm=bert-base-uncased --pooling=average --submodel=12 --seed=0
```


* Mixture of multiple layers
```python
python train.py --lm=bert-base-uncased --pooling=average --submodel=0,1,2,3,4,5,6,7,8,9,10,11,12 --mixing --seed=0
```