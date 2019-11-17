import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StopWordsRemover, RegexTokenizer, HashingTF, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object Tweet {
  def main(args: Array[String]): Unit = {

    if (args.length != 2) {
      println("Usage: spark-submit --class Tweet *.jar InputFile Output File")
      sys.exit(1)
    }

    val inputFile = args(0)
    val outputFile = args(1)

//        val inputFile = "file/Tweets.csv"
//        val outputFile = "file/tweet-output.txt"

    val spark = SparkSession
      .builder()
//      .master("local")  // for local testing
      .appName("Tweet")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // 1. loading - read the data
    val tweets = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputFile)

    // 2. pre-processing
    val tweetsDF = tweets.select("text", "airline_sentiment")
      .filter("text is not null and text != ''")

    val regexTokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("""\W""")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("features")
    val labelIndexer = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("label")

    val pipeline = new Pipeline().setStages(Array(regexTokenizer, stopWordsRemover, hashingTF, labelIndexer))

    val pipelineModel = pipeline.fit(tweetsDF)
    val processedDF = pipelineModel.transform(tweetsDF)

    // 3. model creation
    val lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setMaxIter(10)

    // hyperparameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.005, 0.001))
      .addGrid(lr.elasticNetParam, Array(0.5, 0.3))
      .build()

    val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // 4. model testing & cross validation
    val Array(train, test) = processedDF.randomSplit(Array(0.8, 0.2), seed = 123)
    // training
    val cvModel = cv.fit(train)
    // test result
    val testResDF = cvModel.transform(test)

    val predictionAndLabels = testResDF.select("prediction", "label")
      .rdd.map(x => (x.getAs[Double](0), x.getAs[Double](1)))

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy

    val res = "Accuracy: ".concat(accuracy.toString)

    // 5. write out the output
    spark.sparkContext.parallelize(Seq(res)).saveAsTextFile(outputFile)
  }
}
