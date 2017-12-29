see doc/index.html
## modified crf++
1. add ne scorer for ne segment  
eg "山东高考"，"山东" is an ne, score is   
'''
## 0.9257;0,2,0.9264  
山	b	b/0.926488  
东	e	e/0.926476  
高	o	o/0.99303  
考	o	o/0.999307  
'''
## build  
./configure   
make  

## reference
1. Culotta A, Mccallum A. Confidence estimation for information extraction[C]// Hlt-Naacl 2004: Short Papers. Association for Computational Linguistics, 2004:109-112.
