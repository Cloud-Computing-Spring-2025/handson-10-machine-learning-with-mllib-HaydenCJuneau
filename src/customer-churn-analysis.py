from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline, Model
from pyspark.sql.types import StringType, NumericType

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load dataset
data_path = "/opt/bitnami/spark/Churn/input/customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Identify Columns
label_col = "Churn"
categorical_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType) and f.name != label_col]
numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType) and f.name != label_col]


# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df: DataFrame) -> DataFrame:
    # Fill missing values
    df = df.na.fill(value=0, subset=["TotalCharges"])

    # Encode categorical variables    
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="error")
        for col in categorical_cols
    ]

    label_indexer = StringIndexer(inputCol=label_col, outputCol=f"{label_col}_indexed", handleInvalid="error")

    # One-hot encode indexed features
    encoder = OneHotEncoder(
        inputCols=[f"{col}_indexed" for col in categorical_cols],
        outputCols=[f"{col}_ohe" for col in categorical_cols],
    )
    
    # Assemble features into a single vector
    assembler = VectorAssembler(
        inputCols=[f"{col}_ohe" for col in categorical_cols] + numeric_cols,
        outputCol="features"
    )

    pipeline = Pipeline(stages=[label_indexer] + indexers + [encoder, assembler])

    model = pipeline.fit(df)
    df_transformed = model.transform(df)

    final_df = df_transformed.select("features", f"{label_col}_indexed")
    final_df.show()
   
    return final_df


# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df: DataFrame) -> DataFrame:
    # Split data into training and testing sets
    train, test = df.randomSplit([0.8, 0.2])
    
    # Train logistic regression model
    regression = LogisticRegression(featuresCol="features", labelCol="Churn_indexed")

    pipeline = Pipeline(stages=[regression])

    # Predict and evaluate
    fit = pipeline.fit(train)

    results = fit.transform(test)
    results.show()

    evaluator = BinaryClassificationEvaluator(labelCol="Churn_indexed")
    ROC_integral = evaluator.evaluate(results)
    print(f"Area under ROC curve: {ROC_integral}")

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Churn_indexed",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(results)
    print(f"Accuracy: {accuracy}")
    
    return df


# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df: DataFrame) -> DataFrame:
    selector = ChiSqSelector(
        numTopFeatures=5, 
        featuresCol="features", 
        outputCol="selectedFeatures", 
        labelCol="Churn_indexed",
    )

    result = selector.fit(df).transform(df)

    result.select("selectedFeatures", "Churn_indexed").show()

    return df
   

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df: DataFrame) -> DataFrame:
    # Split data
    train, test = df.randomSplit([0.8, 0.2])

    def cross_evaluate(model, param_grid, name) -> Model:
        evaluator = BinaryClassificationEvaluator(
            labelCol="Churn_indexed", 
            metricName="areaUnderROC",
        )

        validator = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=5,
        )

        # Fit
        print(f"\n\nTuning {name} . . .")
        validator_model = validator.fit(train)
        best_model = validator_model.bestModel

        # Evaluate on Test
        results = best_model.transform(test)
        auc = evaluator.evaluate(results)

        print(f"Best model accuracy (AUC) for {name}: {auc:.2f}")

        return best_model        

    
    # Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol="Churn_indexed")
    lr_param_grid = (ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1, 1.0])
        .addGrid(lr.maxIter, [10, 20])
        .build()
    )

    lr_best = cross_evaluate(lr, lr_param_grid, "LogisticRegression")
    print(f"Best parameters for LogisticRegression: regParam={lr_best._java_obj.getRegParam()} maxIter={lr_best._java_obj.getMaxIter()}")

    
    # Decision Tree Classifier
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="Churn_indexed")
    dt_param_grid = (ParamGridBuilder()
        .addGrid(dt.maxDepth, [5, 10, 15])
        .addGrid(dt.minInstancesPerNode, [1, 5])
        .build()
    )

    dt_best = cross_evaluate(dt, dt_param_grid, "DecisionTreeClassifier")
    print(f"Best parameters for DecisionTreeClassifier: maxDepth={dt_best._java_obj.getMaxDepth()} minInstancesPerNode={dt_best._java_obj.getMinInstancesPerNode()}")


    # Random Forest Classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="Churn_indexed")
    rf_param_grid = (ParamGridBuilder()
        .addGrid(rf.numTrees, [20, 50])
        .addGrid(rf.maxDepth, [5, 10, 15])
        .build()
    )

    rf_best = cross_evaluate(rf, rf_param_grid, "RandomForestClassifier")
    print(f"Best parameters for RandomForestClassifier: numTrees={rf_best._java_obj.getNumTrees()} maxDepth={rf_best._java_obj.getMaxDepth()}")


    # Gradient Boosted Trees (GBT)    
    gbt = GBTClassifier(featuresCol="features", labelCol="Churn_indexed")
    gbt_param_grid = (ParamGridBuilder()
        .addGrid(gbt.maxIter, [10, 20])
        .addGrid(gbt.maxDepth, [5, 10])
        .build()
    )

    gbt_best = cross_evaluate(gbt, gbt_param_grid, "GBTClassifier")
    print(f"Best parameters for GBTClassifier: maxIter={gbt_best._java_obj.getMaxIter()} maxDepth={gbt_best._java_obj.getMaxDepth()}")

    
    return df
    

# Execute tasks
print("- - - Executing Task 1 - - -")
preprocessed_df = preprocess_data(df)

print("- - - Executing Task 2 - - -")
train_logistic_regression_model(preprocessed_df)

print("- - - Executing Task 3 - - -")
feature_selection(preprocessed_df)

print("- - - Executing Task 4 - - -")
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
