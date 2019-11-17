import org.apache.spark.sql.SparkSession

object PageRank {
  def main(args: Array[String]): Unit = {
            if (args.length != 3) {
              println("Usage: spark-submit --class PageRank *.jar InputFile MaxIterations OutputFile")
              sys.exit(1)
            }

            val inputFile = args(0)
            val maxIter = args(1).toInt
            val outputFile = args(2)

//    val inputFile = "file/airport.csv"
//    val maxIter = 10
//    val outputFile = "file/pageRank-output"

    val spark = SparkSession
      .builder()
//      .master("local")  // for local testing
      .appName("PageRank")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    val sc = spark.sparkContext

    val data = sc.textFile(inputFile)
    val header = data.first
    // eliminate the header
    val airports = data.filter(row => row != header)

    // pairs: (ori, dest)
    val pairs = airports.map { row =>
      val line = row.split(",")
      (line(0), line(1))
    }
    // pageRank: (ori, ori_rank)
    var pageRank = pairs.reduceByKey(_ + _).map(x => (x._1, 10.0))
    // outlinks: (ori, #outlinks)
    val outlinks = pairs.map(x => (x._1, 1.0)).reduceByKey(_ + _)
    val totalNum = outlinks.count().toDouble

    // pairs: (ori, dest)
    // oriOutdegree: (ori, (dest, #outlinks))
    val oriOutdegree = pairs.join(outlinks)
    val alpha = 0.15

    for (i <- 1 to maxIter) {
      // join:(ori, (ori_rank, (dest, #outlinks))) => (dest, (ori_rank, #outlinks))
      pageRank = pageRank.join(oriOutdegree).map(x => (x._2._2._1, (x._2._1, x._2._2._2)))
        .map(x => (x._1, x._2._1 / x._2._2))
        .reduceByKey(_ + _)
        .map(x => (x._1, alpha / totalNum + (1 - alpha) * x._2))
    }
    val res = pageRank.sortBy(-_._2)
    res.saveAsTextFile(outputFile)

  }
}