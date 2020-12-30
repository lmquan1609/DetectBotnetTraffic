# Detect Botnet Traffic

The main objective of this project is to detect botnet traffic based on Netflow dataset using various Machine Learning approaches. Specifically, our project attempts to:
* Take Netflow dataset, then classify either normal or attack traffic
* Compare popular Machine Learning methods to figure out the proper model

For this problem, since only Supervised Learning approaches are applied, techniques to avoid the imbalance data, which is popular in Netflow dataset, are required to intelligently search the optimal class weight. Then based on the class weight, Decision Tree and Random Forest are involved in training classifer which is capable of detecting either normal and attack traffic. Tree-based solutions are only applied in this project since most of dataset features are in raw version and only a few preprocessing method are used, e.g., One-hot encoding for categorical features and Min-max scaling for numerical features. After that, once the binary classifier identifies the botnet traffic, one more multi-label classifer is applied to categorize which type of botnet traffic is. The multi-classification model use multi-layer perception with 3 hidden layers.

## Dataset

The data is [The CTU-13 Dataset - A Labeled Dataset with Botnet, Normal and Background traffic](https://www.stratosphereips.org/datasets-ctu13). The dataset has close to 20 million records with 14 features in 13 scenarios which have 7 different botnets. The figure below demonstrates distribution of labels in the Netflows for each scenario
![](https://images.squarespace-cdn.com/content/v1/5a01100f692ebe0459a1859f/1510069535476-AO1EJKKV3EAZLRDOV8ZC/ke17ZwdGBToddI8pDm48kHUhxuZRagHsPIi0KAt0cD5Zw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpzADkEvtZCn5S0bRP0H8A0QE_WBUUJmdqzGcKEaut8wXJDAUESYs10AqzOnGgWAnBA/Table4.jpg)

## Requirements

* python3
* matplotlib
* numpy
* pandas
* seaborn
* keras

## Usage
### Download CTU-13

```
$ bash downloader.sh
```

In case original CTU-13 takes up lots of storage (up to 30GB), folder with only Netflow data can be downloaded via (require `sudo apt-get install unzip`):
```
$ bash downloaderPruned.sh
```

### Data visualization

* All figures in report located in `Figures` folder
```
$ python EDA/basicVisualization.py
```

### Preprocess data
* Generate the Min-max scaling and One-hot Encoder output files to apply in testing set
```
$ python preprocessing/preprocessing.py
$ python preprocessing/genPrepFiles.py
```
* Apply the Min-max scaling and One-hot Encoder output files to testing set
```
$ python preprocessing/applyPrepFiles.py
```

### Model

* All model files are generated in in `final` folder

#### Decision Tree
```
$ python modelDecisionTree.py
```

#### Random Forest
```
$ python modelDecisionTree.py
```

#### Multi-layer perceptron for multi-label classifer
```
$ python modelMLP.py
```