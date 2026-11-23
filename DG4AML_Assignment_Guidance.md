# Understanding the Penguin Assignment

## What this assignment asks you to do

You have 344 records of penguins. Each record describes a penguin using measurements and other information.

The target is sex:

- Male
- Female

The goal is to predict sex without blood tests or capture.

Therefore, the final classifier must use only information that can be collected from photographs or field observation. It must not use:

- Actual sex as an input
- Actual body mass
- Stable nitrogen isotope
- Stable carbon isotope
- Comments
- Species from external sources

The notebook completes four stages:

1. Exploratory Data Analysis and preprocessing
2. Clustering
3. Regression
4. Classification

---

# Basic words

## Dataset

A dataset is a table of observations.

In this dataset:

- One row is one penguin.
- One column is one recorded property.
- 344 rows contain the original penguin observations.
- 15 columns contain the original variables.

For example:

| Penguin | Bill length | Bill depth | Flipper length | Body mass | Sex |
|---|---:|---:|---:|---:|---|
| A | 38.8 millimetres | 17.6 millimetres | 191 millimetres | 3275 grams | Female |
| B | 34.6 millimetres | 21.1 millimetres | 198 millimetres | 4400 grams | Male |

## Feature

A feature is an input column used to predict an answer.

In final classification, the original features are:

- Culmen length
- Culmen depth
- Flipper length
- Island
- Clutch completion

## Target

A target is the value being predicted.

Targets in this project:

- Body mass during regression
- Sex during classification

A target must not be placed in the input feature table.

## Engineered feature

An engineered feature is a new input created by a model or a calculation.

This project creates two engineered features:

- A cluster label
- A predicted body-mass value

## Training data

Training data is used to fit the model.

The model compares the training features with the training answers and learns a relationship.

## Test data

Test data is held back and used to measure performance.

The test data is not used to:

- Fit the cluster model
- Fit imputation values
- Fit scaling values
- Select classification settings
- Train the final classifiers

## Cross-validation

Cross-validation divides the training data into several equal folds.

The model:

1. Fits on some folds.
2. Validates on the remaining fold.
3. Repeats until every fold has been validation once.
4. Calculates the average result.

This gives a more reliable estimate than using one validation partition.

## Data leakage

Data leakage happens when information from test or answer data enters the training process.

Leakage makes results look too good. The model will appear accurate but may fail on genuinely new penguins.

---

# Exact dataset information

The original CSV has:

- 344 rows
- 15 columns
- 168 male rows
- 165 female rows
- 11 rows with missing sex
- 333 rows with known sex

The outer split produces:

- 249 training rows
- 84 test rows
- 126 males and 123 females in training
- 42 males and 42 females in testing

The test set is balanced by sex because stratification preserves the class proportion.

---

# Meaning of every important column

## Culmen length

The culmen is the upper ridge of the bill.

Culmen length is the bill length in millimetres.

It is permitted because a calibrated photograph can estimate it.

It is used in:

- K-means clustering
- Body-mass regression
- Final sex classification

## Culmen depth

Culmen depth is the vertical height or depth of the bill in millimetres.

It is permitted for the same reason as culmen length.

It is used in the same three stages.

## Flipper length

Flipper length is the length of the penguin flipper in millimetres.

It can be estimated from a field photograph.

It is used in clustering, regression and final classification.

## Island

Island is a category:

- Dream
- Torgersen
- Biscoe

It is permitted because location does not require capture or blood sampling.

It is converted to binary indicator columns by one-hot encoding.

## Clutch completion

Clutch completion records whether a complete clutch was observed.

It is permitted and available from field observation.

It is converted to binary indicator columns.

## Body mass

Body mass is measured in grams.

It is not permitted as a final classification feature because weighing requires capture.

It is used here only as the answer in regression.

The regression prediction is allowed as an engineered classification feature.

## Sex

Sex is the final classification target.

It is:

- Retained for supervised training and testing
- Removed from the final input feature table
- Never used to generate a clustering answer

## Stable nitrogen isotope

Delta 15 N requires sampled tissue or blood.

It is not used as a feature.

## Stable carbon isotope

Delta 13 C requires sampled tissue or blood.

It is not used as a feature.

## Comments

Comments are mostly blank free-text notes.

They are not used because they are unreliable and not consistently available.

## Individual ID

Individual ID identifies a penguin sample.

It is not used because it does not describe a new penguin's morphology or context.

It could encourage memorisation rather than learning patterns.

## Region, stage and date

These columns do not provide a required permitted feature for this project.

They are not used in the final feature table.

---

# Stage 1: Exploratory Data Analysis and preprocessing

## The first data finding

The first finding is missing data.

Relevant missing counts are:

- 11 sex values
- 2 culmen length values
- 2 culmen depth values
- 2 flipper length values
- 2 body-mass values
- 14 stable nitrogen isotope values
- 13 stable carbon isotope values
- About 290 comment values

## How missing values are handled

Sexless rows are removed for the supervised sex task:

```python
model_df = df.dropna(subset=["Sex"]).copy()
```

This is valid because sex is the answer. A row cannot teach or test classification when the answer is unknown.

Missing permitted measurements are retained and replaced with training-derived medians:

```python
SimpleImputer(strategy="median")
```

For example, if the training median bill depth is 17.3 millimetres, a missing bill-depth value becomes 17.3 millimetres.

The median is less sensitive to extreme values than the mean.

The stable-isotope and comment columns are not used.

## The second data finding

The boxplots show that male and female distributions overlap.

The measurements have useful differences, but no single measurement perfectly separates the sexes.

The correlation heatmap shows relationships among the permitted measurements and body mass.

This finding influences the following choices:

- Scale numerical measurements for logistic regression.
- Include a regularised linear model.
- Test a non-linear random forest model.
- Retain several related variables rather than choosing only one.

## One-hot encoding

Categorical text cannot be directly used as a number.

For three islands:

| Island | Dream | Torgersen | Biscoe |
|---|---:|---:|---:|
| Dream | 1 | 0 | 0 |
| Torgersen | 0 | 1 | 0 |
| Biscoe | 0 | 0 | 1 |

This creates numeric indicator columns.

It does not invent an incorrect order such as Dream 1, Torgersen 2 and Biscoe 3.

## Standard scaling

Standard scaling converts a numeric column using:

\[
z=\frac{x-\text{training mean}}{\text{training standard deviation}}
\]

After scaling:

- The mean is close to zero.
- The standard deviation is close to one.

Scaling is used because:

- Culmen depth is in a different range from flipper length.
- Logistic regression uses coefficient optimisation.
- K-means uses distances.

Scaling is fitted only on training rows.

## The outer train-test split

The code is:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

This means:

- 75 percent of labelled rows go to training.
- 25 percent go to the held-out test set.
- Stratification preserves the male/female proportion.
- Random state 42 makes the split reproducible.

---

# Stage 2: Clustering

## Purpose

Clustering creates the first engineered feature.

It uses only:

- Culmen length
- Culmen depth
- Flipper length

It does not use sex, body mass, stable isotopes or species.

## Preprocessing for clustering

The clustering pipeline:

```python
cluster_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

The steps are:

1. Replace missing measurements using the training median.
2. Scale measurements using training mean and standard deviation.
3. Apply K-means.

The cluster model is fitted on outer-training rows.

Test rows are only transformed and assigned.

## K-means

K-means is an unsupervised algorithm.

Unsupervised means it has no answer column.

It repeatedly:

1. Starts with cluster centres.
2. Assigns each penguin to the nearest centre.
3. Recalculates every centre.
4. Repeats until assignments become stable.

The output is a cluster number such as 0 or 1.

## Choosing the cluster count

The notebook tests:

- 2 clusters
- 3 clusters
- 4 clusters
- 5 clusters
- 6 clusters

## Silhouette score

The silhouette score compares:

- Distance from a penguin to its own cluster
- Distance from that penguin to the nearest other cluster

The measured scores are:

| Cluster count | Silhouette score |
|---:|---:|
| 2 | 0.508 |
| 3 | 0.491 |
| 4 | 0.446 |
| 5 | 0.365 |
| 6 | 0.373 |

The highest score is 2.

Therefore, K-means uses two clusters.

This does not prove that species exists or that a cluster equals species.

It proves only that the data supported a two-group measurement structure.

## Stability

The selected model was tested on ten random 80 percent subsamples.

The mean adjusted Rand index was 1.0.

Adjusted Rand index measures whether repeated cluster assignments agree.

The value 1.0 means the tested subsample assignments were highly consistent.

## Cluster integration

Training cluster labels are created by:

```python
cluster_train = cluster_model.fit_predict(X_cluster_train)
```

Test cluster labels are created by:

```python
cluster_test = cluster_model.predict(X_cluster_test)
```

The test data does not create or alter the cluster centres.

Training cluster counts:

- Cluster 0: 83 rows
- Cluster 1: 166 rows

Test cluster counts:

- Cluster 0: 37 rows
- Cluster 1: 47 rows

---

# Stage 3: Regression

## Purpose

Regression predicts a number.

The target is body mass in grams.

The inputs are:

- Culmen length
- Culmen depth
- Flipper length
- Island
- Clutch completion
- Cluster label

Actual body mass is not used as a final classifier feature.

## Random forest regression

A random forest is a collection of many decision trees.

A decision tree asks threshold questions such as:

```text
Is flipper length above 205 millimetres?

Yes: predict a larger body mass
No: predict a smaller body mass
```

Many trees produce many estimates.

The forest averages those estimates.

## Hyperparameters

Hyperparameters are settings chosen before final fitting.

Grid search compares:

- 200 or 400 trees
- Unlimited depth or depth 8
- Minimum leaf size 1 or 3

The selected settings are:

- 200 trees
- Maximum depth 8
- Minimum leaf size 3

## Held-out results

| Metric | Mean-mass baseline | Random forest |
|---|---:|---:|
| Mean Absolute Error | 789.050 grams | 278.080 grams |
| Root Mean Squared Error | 910.367 grams | 366.617 grams |
| R-squared | -0.023 | 0.834 |

### Mean Absolute Error

Mean Absolute Error is the average absolute distance between actual and predicted body mass.

The forest typically missed body mass by about 278 grams.

### Root Mean Squared Error

Root Mean Squared Error gives larger errors a stronger penalty.

The forest value is 366.617 grams.

### R-squared

R-squared measures explained held-out variation.

The result 0.834 means the model explained approximately 83.4 percent of held-out body-mass variation.

The baseline value -0.023 is below the simple test-target mean reference.

## Residuals

A residual is:

\[
\text{Residual}=\text{Actual body mass}-\text{Predicted body mass}
\]

Residual plots check whether errors:

- Increase with larger predictions
- Cluster around zero
- Show curved patterns
- Have unequal spread

## Out-of-fold predicted body mass

The training feature uses cross-validation predictions:

```python
oof_mass_train = cross_val_predict(...)
```

For every training row:

1. That row is excluded from one fit.
2. The remaining training rows fit a regression model.
3. The model predicts the excluded row.
4. The prediction is stored as a new feature.

This prevents a row from contributing both its answer and a self-generated answer.

Test body-mass predictions use a regression model fitted on all outer-training rows:

```python
mass_test_pred_full = reg_search.predict(X_test_clusters[reg_features])
```

Actual test body mass is never used.

---

# Stage 4: Classification

## Final feature set

The final classification features are:

- Culmen length
- Culmen depth
- Flipper length
- Island
- Clutch completion
- Cluster label
- Predicted body mass

The following are excluded:

- Sex
- Actual body mass
- Stable nitrogen isotope
- Stable carbon isotope
- Comments
- Species

## Classification approach 1: Logistic regression

Logistic regression predicts probability.

It calculates:

\[
P(\text{Male})=\frac{1}{1+e^{-z}}
\]

The probability is between zero and one.

The default threshold is 0.50:

- 0.50 or above predicts male.
- Below 0.50 predicts female.

The tuned settings are:

- C value 10
- lbfgs solver
- Balanced class weights

C controls regularisation strength.

Smaller C applies stronger regularisation.

Larger C allows coefficients to fit training data more closely.

## Classification approach 2: Random forest

A random forest classifier trains many decision trees.

Each tree produces a vote or probability.

The final result combines every tree.

This is qualitatively different from logistic regression because it can model:

- Thresholds
- Non-linear boundaries
- Interactions

The tuned settings are:

- 500 trees
- Maximum depth 8
- Minimum leaf size 1
- Balanced class weights

## Held-out comparison

| Metric | Logistic regression | Random forest |
|---|---:|---:|
| Accuracy | 85.7 percent | 82.1 percent |
| Balanced accuracy | 85.7 percent | 82.1 percent |
| Macro precision | 85.8 percent | 82.3 percent |
| Macro recall | 85.7 percent | 82.1 percent |
| Macro F1 | 85.7 percent | 82.1 percent |
| Receiver Operating Characteristic Area Under the Curve | 93.2 percent | 90.6 percent |

## Accuracy

Accuracy is the proportion of correct predictions:

\[
\text{Accuracy}=\frac{\text{Correct predictions}}{\text{All predictions}}
\]

## Balanced accuracy

Balanced accuracy gives male and female equal weight:

\[
\text{Balanced accuracy}
=\frac{\text{Male correct rate}+\text{Female correct rate}}{2}
\]

This is the main model-selection measure.

## Confusion matrix

The confusion matrix separates four results:

|  | Predicted male | Predicted female |
|---|---|---|
| Actual male | Correct male | Incorrect female |
| Actual female | Incorrect male | Correct female |

The matrix shows class-specific errors.

Accuracy alone cannot show which sex is being misclassified.

## Precision

Precision answers:

"When the model predicts male, how often is the result actually male?"

## Recall

Recall answers:

"Of all actual males, how many did the model identify?"

## F1

F1 combines precision and recall.

Macro F1 calculates F1 for both classes separately and averages them.

## Receiver Operating Characteristic Area Under the Curve

This metric measures probability ranking across all thresholds.

A result of 0.932 means that, for a randomly selected male-female pair, the model ranks the male as more likely to be male 93.2 percent of the time.

It does not mean 93.2 percent accuracy.

## Final choice

Logistic regression is selected because it has the higher held-out balanced accuracy and better results for every reported metric.

It is also easier to explain and deploy.

The conclusion is based on observed held-out values.

---

# Complete leakage-safe workflow

```text
Original CSV
    |
    v
Remove rows without sex for supervised classification
    |
    v
75 percent train / 25 percent test outer split
    |
    v
Fit imputation and scaling on training only
    |
    v
Fit K-means on outer-training permitted measurements
    |
    v
Add cluster label to training and test data
    |
    v
Fit body-mass random forest
    |
    v
Create out-of-fold mass values for training rows
    |
    v
Predict test mass using the outer-training-fitted regression model
    |
    v
Compare logistic regression and random forest
    |
    v
Evaluate both on held-out sex labels
```

---

# Professor questions

## Why is sex classification the problem?

Sex is a category rather than a number.

The task is to place each penguin into male or female.

## Why is species not used?

The species column is absent and external reconstruction is prohibited.

## Why was the data split first?

The split prevents test data from influencing fitted statistics, cluster centres, hyperparameter choice or model coefficients.

## Why is median imputation used?

It preserves incomplete rows and replaces missing values with a typical training value.

## Why is scaling used?

Scaling allows logistic regression and K-means to compare variables without raw units dominating.

## Why choose K-means?

K-means is suitable for continuous permitted measurements and creates a fixed cluster feature.

## Why test several cluster counts?

There is no given correct cluster count.

Silhouette scores allow an evidence-based choice.

## Does the cluster label represent species?

No.

It represents measurement-based group structure.

Species must not be claimed from it.

## Why use random forest for regression?

It can learn non-linear morphology and body-mass relationships.

It was compared against a mean-mass baseline.

## Why use out-of-fold predictions?

They stop training body-mass predictions from seeing their own body-mass target.

## Why use two classification methods?

Logistic regression is linear and interpretable.

Random forest is non-linear and interaction-based.

The assignment requires qualitatively different approaches.

## Why choose logistic regression?

It has the higher balanced accuracy and the higher Receiver Operating Characteristic Area Under the Curve.

It is simpler and more transparent.

## Are the results enough for deployment?

No.

The project shows the method works on this hold-out sample.

It needs external field validation before operational use.

---

# Honest limitations

- The test result comes from one outer split.
- The dataset has a moderate number of labelled rows.
- Male and female measurements overlap.
- The clustering stability test uses subsamples, not unseen islands or years.
- Sex may be influenced by biology not contained in the permitted variables.
- The model has not been externally validated.

These limitations should be stated rather than hidden.

---

# One-minute answer

"I built an end-to-end penguin-sex pipeline. I used a 75/25 stratified outer split to keep test sex answers separate. Imputation and scaling were fitted inside pipelines using training data only. K-means used three permitted measurements and selected two clusters using silhouette scores. Random forest regression estimated body mass, and training estimates were generated out of fold to prevent leakage. I compared logistic regression with random forest and selected logistic regression because it achieved 85.7 percent held-out balanced accuracy, compared with 82.1 percent for random forest."
