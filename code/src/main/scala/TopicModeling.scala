import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable


object TopicModeling {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: spark-submit --class TopicModeling *.jar InputFile OutputFile")
      sys.exit(1)
    }

    val inputFile = args(0)
    val outputFile = args(1)

//        val inputFile = "file/Sherlock_Holmes.txt"
//        val outputFile = "file/topicModeling-output"

    val spark = SparkSession
      .builder()
//      .master("local") // for local testing
      .appName("TopicModeling")
      .getOrCreate()

    val sc = spark.sparkContext

    val corpus = sc.textFile(inputFile)
    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet
    val tokenized: RDD[Seq[String]] = corpus.map(_.toLowerCase.split("\\s"))
      .map(_.filter(_.length > 3).filter(w => !stopWordSet.contains(w)).filter(_.forall(java.lang.Character.isLetter)))

    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    //   vocabArray: Chosen vocab (removing common terms)
    val numStopwords = 20
    val vocabArray = termCounts.takeRight(termCounts.length - numStopwords)

    //   vocab: Map term -> term index
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.map(x => (x._1._1, x._2)).toMap

    // Convert documents into term count vectors
    val documents: RDD[(Long, Vector)] = tokenized.zipWithIndex.map { case (tokens, id) =>
      val counts = new mutable.HashMap[Int, Double]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val idx = vocab(term)
          counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
        }
      }
      (id, Vectors.sparse(vocab.size, counts.toSeq))
    }

    // Set LDA parameters
    val numTopics = 5
    val lda = new LDA().setK(numTopics).setMaxIterations(10)

    val ldaModel = lda.run(documents)

    // Print topics, showing top-weighted 10 terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    var res = ""

    topicIndices.zipWithIndex.foreach { case ((terms, termWeights), i) =>
      val idx = i + 1
      res = res.concat(s"TOPIC $idx:\n")
      terms.zip(termWeights).foreach { case (term, weight) =>
        res = res.concat(s"${vocabArray(term.toInt)._1}\t$weight\n")
      }
      res = res.concat("\n")
    }
    val result = sc.parallelize(Seq(res))
    result.saveAsTextFile(outputFile)
  }
}
