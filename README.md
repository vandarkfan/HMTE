<h1 align="center">
  HMTE
</h1>

<h4 align="center">HMTE: Mamba-Transformer Representation Learning for Knowledge Hypergraph Completion</h4>

<h2 align="center">
  Overview of HMTE
  <img align="center"  src="overview.png" alt="...">
</h2>

This paper has been submitted to the WWW 2025.

### Dependencies

- python            3.10.13
- torch             2.1.1+cu118
- mamba-ssm         2.2.2
- numpy             1.26.3
- transformers      4.44.1
- The installation of mamba-ssm package can refer to https://github.com/state-spaces/mamba. 

### Dataset:

- We use JF17K, WikiPeople, and FB-AUTO dataset for knowledge hypergraph link prediction. 
- All the preprocessed data are included in the `./data` directory.

### Results:
The results are:

|  Dataset   |  MRR  |  H@1  |  H@3  | H@10  |
| :--------: | :---: | :---: | :---: | :---: |
|   JF17K    | 0.597 | 0.519 | 0.634 | 0.747 |
| WikiPeople | 0.450 | 0.372 | 0.489 | 0.585 |
|  FB-AUTO   | 0.874 | 0.856 | 0.884 | 0.906 |

## How to Run
```
python main-JF17K.py                 ## JF17K dataset
python main-WikiPeople.py            ## WikiPeople dataset
python main-FB-AUTO.py               ## FB-AUTO dataset
```

