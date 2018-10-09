# The Data Science of Marketing

### Introduction 
#### Four Critera for Quality Data
- Must be reliable
- Should be in raw format
- Should be well documented
- Must be well organized

#### Preapre Your Data:
ETL - Extract, Transform, Load

#### Extract Data From
-	CRM Platform
-	Marketing Automation Platform
-	Paid Search Campaign Platform
- They all offer API - Application Programming Interface (Make you extract data automatically)
- Three Steps for Extraction
	- **Define**: Define what data you need
	- **Proceure**: Determine what data you can obtain that aligns with that definition and put a process in place for procurement
	- **Store**：Store the data into data warehouse


----------
### Tools 
- R Studio
- Python
- Tableau


----------
### Data, Exploartory Analysis, and Performance Analysis 

TMTMTM = the metrics that matter the most

#### Exploratory Analysis with R/python
- Build a heatmap to get a quick sense of the data, we can use this heatmap to identify some trail hits for deeper analysis or actionable insights. 

```r
# import csv file 
myExploratoryData<-read.csv('~/Exercise_Files/02_02/exploratory-r.csv')
# get a quick snapshot of your data
head(myExploratoryData)
hist(myExploratoryData$cpa)
# shift the names to each row
row.names(myExploratoryData)<-myExploratoryData$keyword
# review that transformation 
head(myExploratoryData)
# transform into a matrix 
myDataMatrix<-data.matrix(myExploratoryData)
# generate our heatmap
heatmap(myDataMatrix,Rowv = NA,Colv = NA,scale = 'column')
```

```python
# import our packages
import pandas as pd
%matplotlib inline
# connect to our data
myExploratoryData=pd.read_csv('~/Exercise_Files/02_03/exploratory-py.csv')
# see a summary of our data
myExploratoryData.head()
# visualize our data
import seaborn as sns
sns.kdeplot(myExploratoryData.cpa)
# visualize our data with additional detail 
sns.distplot(myExploratoryData.cpa)
# pivot the data 
myETLData=myExploratoryData.pivot(index='keyword',columns='impressions',values='cpa')
# see a summary of our pivot 
myETLData
```
Python and R both offer a large range of data science options and capabilities for statistical modeling when comparing to Tableau. 
- **R**
	- Great for addressing most data analysis need
	- Requires significant time investment
- **Python**
	- Well tested and continuously evolving
	- Jupyter can interpret many languages, it has a versatile environment

- **Tableau**
	- Small learnig curve
	- Not as powerful as python and R in modeling 

----------
### Inference and Regression Analysis
For example, More Marketing, More Engagement
#### Regression with R
```r
# Connect to our data
myRegressionData <- read.csv('~/Exercise_Files/02_02/exploratory-r.csv')
# Plot our data (broadcast & sales)
plot(myRegressionData$BROADCAST,myRegressionData$NET.SALES)
# Fit a line
myLm<-lm(myRegressionData$NET.SALES~myRegressionData$BROADCAST)
# Visualize the line
lines(myRegressionData$BROADCAST,myLm$fitted.values)
# Show our coefficients 
myLm$coeff
```

#### Regression with python

```python
# Bring our packages in 
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
%matplotlib inline

# Connect to our data
myRegressionData = pd.read_csv('~/Exercise_Files/02_02/exploratory-r.csv')

# View a snapshot of our data 
myRegressionData.head(5)

# Plot the data
myRegressionData.plot(kind='scatter',x='broadcast',y='sales')

# Calculate r-squared
slope, intercept, r_value, p_value, std_err = stats.linregress(myRegressionData.broadcast,myRegressionData.sales)

# Print the r-squared value
print('r-squared value is:', r_value**2)

# Model OLS to generate coefficients
myLinearModel=smf.ols(formula='sales~broadcast',data=myRegressionData).fit()

# Output our coefficient 
myLinearModel.params
```


----------
### Predictions
#### Prediction with R (Classification Decision Tree)

```r
# Connect to our data
myPredictionData <- read.csv('~/Exercise_Files/04_02/prediction-r.csv')
# Sum our classifications so we can see them 
table(myPredictionData$sales.classification)
# Output our column names for easy reference 
names(myPredictionData)
# Install our algorithm (the tree package)
install.packages('tree')
# Bring our newly installed algorithm into our library of packages
library(tree)
# Configure our algorithm to create our tree
myDecisionTree <- tree(sales.classification ~ capita + drive.by.traffic + complimentary.establishments + competition + weather + unemployment.rate + var1 + var2 + var3, data=myPredictionData)
# Plot our tree so we can see the algorithms output
plot(myDecisionTree)
# Label our tree
text(myDecisionTree)
# Prune our tree
MyPruneTree<-prune.tree(myDecisionTree,best = 3)
# Plot our pruned tree
plot(MyPruneTree)
# Label our pruned tree 
text(MyPruneTree)
```
#### Prediction with Python (Classification Decision Tree)

```python
# Establish the functionality for our assessment by bringing in the right packages
# Make sure to install these prior to mounting the packages 
# i.e. $ pip install pydotplus
# & visit http://www.graphviz.org/Download_macos.php

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pydotplus as pdot
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export
from sklearn.cross_validation import train_test_split
%matplotlib inline
# Connect to the data source
myPredictionData = pd.read_csv("~/Exercise_Files/04_03/prediction-py.csv")
# Set up our cross validation function
feature_cols = ['capita', 'competition', 'weather', 'var1', 'var2', 'var3' ]
train_X, test_X, train_y, test_y = train_test_split( myPredictionData[feature_cols],                            myPredictionData['sales_classification'])
# Specify the number of branches for our tree
clf_tree=DecisionTreeClassifier(max_depth=8)
# Fit our training data to the x and to the y
clf_tree.fit(train_X,train_y)
# Apply our test data to our model 
tree_predict=clf_tree.predict(test_X)
# Visualize our tree
export_graphviz( clf_tree,
                out_file = "model_tree.odt",
                feature_names = train_X.columns )
model_tree_graph = pdot.graphviz.graph_from_dot_file( 'model_tree.odt' )
model_tree_graph.write_jpg( 'model_tree.jpg' )
from IPython.display import Image
Image(filename='model_tree.jpg')

```


----------
### Clustering
- Cluster Analysis Uses
	- Consumer Segmentation
	- Identify Brand Lift
	- identify what people are responding to 

Target right customers to drive revenue: We havve to determine where our marketing will provide value

#### Cluster Analysis With R

```r
# Connect to our case study data
myClusterData <- read.csv("~/Exercise_Files/05_02/cluster-r.csv")
# Review our data
head(myClusterData)
# Standardize the data
myClusterDataStandardized<-scale(myClusterData[-1]) # Remove the first column data
head(myClusterDataStandardized)
# Run kmeans on our standardized data
ourGroups<-kmeans(myClusterDataStandardized,3)
# Load in our cluster library 
library(cluster)
# Visualize our clusters
clusplot(myClusterDataStandardized,ourGroups$cluster)
# Summarize our data
ourGroups$size
```

#### Cluster Analysis with Python

```python
# Load in our packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
%matplotlib inline
myClusterData = pd.read_csv("~/Exercise_Files/05_03/cluster-py.csv")
# Pivot our data to work as an array 
a=myClusterData.b1
b=myClusterData.b2
X = np.column_stack(a,b) # Generate an array by combining two lists
# Assign the value of n clusters / run the algorithm / assign centroids / label our group names 
myGroups=KMeans(n_clusters=3)
myGroups.fit(X)
centroids=myGroups.cluster_centers_
labels=myGroups.labels_

# Set up our color palette
colors = ["b.","g.","r.","c.","m."]
# Plot each point
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
# Generate the view
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5)
plt.show()
```


----------


###  Conjoint Analyst

#### Conjoint Analysis with R
```r
myConjointData <- read.csv("~/Exercise_Files/06_02/conjoint-r.csv")
myConjointDataProfilesMatrix <- read.csv("~/Exercise_Files/06_02/conjoint-r-profiles-matrix.csv")
myConjointDataLevelNames <- read.csv("~/Exercise_Files/06_02/conjoint-r-level-names.csv")

library(conjoint)
# Model some of our data
caUtilities(y=myConjointData[1,],x=myConjointDataProfilesMatrix,z=myConjointDataLevelNames)
# Model all of our data
caUtilities(y=myConjointData,x=myConjointDataProfilesMatrix,z=myConjointDataLevelNames)
```
![Alt text](./Rplot.png)

#### Conjoint Analysis with Python

```python
# Load in our packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
myConjointData = pd.read_csv('~/Exercise_Files/06_03/conjoint-py.csv')
# Update the names of our vectors
names = {"Rank":"Rank", "P1": "PhotoF1","P2": "PhotoF2", "U1": "Ux1", 
         "U2": "Ux2", "D1":"SpecialSauce1", 
         "D2":"SpecialSauce2", "D3":"SpecialSauce3"}
# Apply those new names
myConjointData.rename(columns=names,inplace=True)

# Assign our test data to the x
X = myConjointData[['PhotoF1', 'PhotoF2', 'Ux1', 'Ux2', 'SpecialSauce1', \
                    'SpecialSauce2', 'SpecialSauce3']]
                 
# Assign a constant-benchmark
X=sm.add_constant(X)
# Assign our resulting test data to the y
Y=myConjointData.Rank
# Perform a linear regression model 
myLinearRegressionForConjoint=sm.OLS(Y,X).fit()
myLinearRegressionForConjoint.summary()

# Normalize values for each feature 
raw = [3.67,3.05,2.72] #Top three features
norm = [float(i)/sum(raw) for i in raw]
norm

# Graph our winning product features
labels = 'Special Sauce Feature Three', 'User Experience Feature One', 'Photo Feature One'
sizes = [39, 32, 29] #Raw 
colors = ['yellowgreen', 'mediumpurple', 'lightskyblue'] 
explode = (0, 0.1, 0)    
plt.pie(sizes,              
        explode=explode,   
        labels=labels,      
        colors=colors,      
        autopct='%1.1f%%',  
        shadow=True,        
        startangle=70       
        )
plt.axis('equal')

```


----------
### Practices For Data Driven Marketing

#### Agile Marketing
An iterative workflow implementation focused on speed, responsiveness, and maximizing performance. 

#### Marketing Campaign Test - MVC (Minimum Viable Campaign)
-	MVC Road Map: Include a hypothesis, stated objectives, requirements for data, a resource plan
-	Execute the campaign
-	Analyze the data 

#### Stakeholder Alignment
Get more people involved, get them align with your data driven marketing. 