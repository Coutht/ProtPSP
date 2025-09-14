# ProtPSP

ProtPSP provides the prediction of general and kinase-specific phosphorylation site using deep leraning. Developer is Guan bifeng from College of Computer and Data Science, Fuzhou University of China.

# Requirement
python == 3.5, 3.6 or 3.7

keras == 2.1.2

tensorflow == 1.14.0

numpy >= 1.8.0

backend == tensorflow

# data and model wieght download 
you can download our data and model weight at 通过网盘分享的文件：protpsp
链接: https://pan.baidu.com/s/1Jtf0RnZnj30TDVBo0Bg8ag?pwd=1111 提取码: 1111
ProtT5 model weight can download at https://huggingface.co/Rostlab/prot_t5_xl_uniref50


# Predict For Your Test Data
cd to the ProtPSP fold. set you owner filename and predict savepath and residue in predict.py.
before predict you must ensure you have feature extract from ProtT5 as embedding file.

If you want to predict general site, taking S/T site as a example, run:

'''python predict.py'''

Output file includes three columns, position, residue type and score. The value range of score is [0, 1], with values closer to 1 indicating the site is more likely to be phosphorylated.

# Train For Your own Data
you should process you data as csv file like 

<img width="703" height="377" alt="image" src="https://github.com/user-attachments/assets/acca4469-bd0e-43bd-919a-fe16fe30b52c" />

from left to right is label, uniprot_id, length, residue-position, sequence respectively.
before predict you must ensure you have feature extract from ProtT5 as embedding file.

if you want to train for ST site, you can cd train_1.py, then set you owner train_file_name in train_1.py,then run 

'''python train_1.py'''


# Contact
Please contact me if you have any help: 3236522157@qq.com

