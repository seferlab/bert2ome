BERT2OME

4 different datasets are used for detecting 2'-O-methylation sites in the given RNA sequences.
- RMBase_800.xlsx (named as "Human 2 Dataset" in our paper) 
- H. sapiens Dataset (named as "Human 1 Dataset" in our paper)
- S. cerevisiae Dataset
- M. musculus Dataset

Last 3 datasets can be downloaded from the following website: http://lab.malab.cn/~acy/PTM_data/RNADataset.html
Each file should be run seperately. 

MACHINE LEARNING PART:
- BaseModels_*.py files is used for evaluating the prediction performance according to Decision Tree, Random Forest, XGBoost and SVM models.
While designing our base models, one-hot conversion is used for converting each RNA sequence into vector format and then models are trained accordingly. 

DEEP LEARNING PART: 

In this section, we use one of the well-known transformer base model BERT, for converting given RNA sequences into vector embeddings format.

- VectorEmbeddingsCreation_DNA.py file is used for converting RMBase dataset (named as Human 1 Dataset in the paper) into vector embeddings format by using BERT.
- VectorEmbeddingsCreation_RNA.py file is used for converting H. sapiens Dataset (named as Human 2 dataset in the paper) into vector embeddings format by using BERT.
- VectorEmbeddingsCreation_M.py file is used for converting M. musculus dataset into vector embeddings format by using BERT.
- VectorEmbeddingsCreation_S.py file is used for converting S. cerevisiae dataset into vector embeddings format by using BERT.

After the previous conversion, following files are generated:

- RNAEMBEDDINGSX.npy
- RNAEMBEDDINGSY.npy

- DNAEMBEDDINGSX.npy
- DNAEMBEDDINGSY.npy

- SEMBEDDINGSX.npy
- SEMBEDDINGSY.npy

- MEMBEDDINGSX.npy
- MEMBEDDINGSY.npy

*EMBEDDINGSX files: Vector embeddings for RNA sequences.

*EMBEDDINGSY files: Vector embeddings for RNA labels.

Random Forest and XGBoost models are fed with these vector embeddings and compared with the previous approach (training models with one-hot formatted RNA sequences). Following files are used for this part:

- BERT+RFandXGB_RNA.py
- BERT+RFandXGB_DNA.py
- BERT+RFandXGB_S.py
- BERT+RFandXGB_M.py

We implemented BERT+1D CNN model for different species, these are the coding files:

- BERT+1DCNN_RNA.py
- BERT+1DCNN_DNA.py
- BERT+1DCNN_S.py
- BERT+1DCNN_M.py

You can see our proposed method BERT2OME, and how we modify it in the following:

- BERT+2DCNN_Human1.py
- BERT+2DCNN_Human2.py
- BERT+2DCNN_S.py
- BERT+2DCNN_M.py
- BERT+2DCNN+HyperparameterTuning+ChemicalProperties_DNA.py
- BERT+2DCNN+HyperparameterTuning_DNA.py
- BERT+2DCNN+HyperparameterTuning_DNA.py
- BERT+2DCNN_RNA_KFold.py
- BERT+2DCNN_DNA_KFold.py
















