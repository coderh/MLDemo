package logisticRegression

import scala.util.Random
import scala.math.{exp, sqrt}
import spark.{RDD, SparkContext}
import spark.util.Vector

import scalala.tensor.dense._
import scalala.library.Plotting._

/**
 * Created with IntelliJ IDEA.
 * User: coderh
 * Date: 9/12/13
 * Time: 3:36 PM
 *
 * run logistic regression example on spark local mode.
 */

object logReg {

  // Number of data points
  val M = 10000

  // Number of dimensions
  val D = 2

  // spark context settings
  val sc = new SparkContext("local[2]", "SparkLR", System.getenv("SPARK_HOME"), Seq(System.getenv("SPARK_EXAMPLES_JAR")), Map[String, String]())

  case class DataPoint(x: Vector, y: Double)

  def generateData = {
    def generatePoint(i: Int) = {
      val x = Vector(D, _ => Random.nextFloat * 3)
      DataPoint(x, if (x(0) + x(1) < 3) 0 else 1)
    }
    Array.tabulate(M)(generatePoint)
  }

  def vectorDivision(v1: Vector, v2: Vector) = Vector(v1.elements zip v2.elements map (p => p._1 / p._2))

  def scale(points: RDD[DataPoint]) = {
    val nbpts = points.count
    val xs = points.map(_.x)
    val x_mean = xs.reduce(_ + _) / nbpts
    val x_centralize = xs.map(_ - x_mean)
    val variance = x_centralize.map(x => Vector(x.elements.map(v => v * v))).reduce(_ + _) / nbpts
    val std_dev_vect = Vector(variance.elements.map(v => sqrt(v)))
    points.map(pt => DataPoint(vectorDivision(pt.x - x_mean, std_dev_vect), pt.y))
  }

  def split[T: ClassManifest](data: RDD[T], p: Double, seed: Long = System.currentTimeMillis): (RDD[T], RDD[T]) = {
    val rand = new java.util.Random(seed)
    val partitionSeeds = data.partitions.map(partition => rand.nextLong)
    val temp = data.mapPartitionsWithIndex((index, iter) => {
      val partitionRand = new java.util.Random(partitionSeeds(index))
      iter.map(x => (x, partitionRand.nextDouble))
    })
    (temp.filter(_._2 <= p).map(_._1), temp.filter(_._2 > p).map(_._1))
  }

  def learningCurvePlot(iterNb: Array[Int], cost: Array[Double]): Unit = {
    plot.hold = true
    plot(iterNb, cost)
    ylim(0.0, cost.max * 1.1)
    xlabel("Iteration")
    ylabel("Cost")
  }

  def localLogReg(initWeight: Vector, iteration: Int, learningRate: Double, testThreshold: Double) = {
    // data loading
    val numSlices = 2
    val points = sc.parallelize(generateData, numSlices).cache()

    // feature scaling
    val scaledPoints = scale(points)
    val generalizedPoints = scaledPoints.map(pt => DataPoint(Vector(1.0 +: pt.x.elements), pt.y))

    // Data Division
    val (trainingSet, remainingSet) = split(generalizedPoints, 0.6)
    val (validationSet, testSet) = split(remainingSet, 0.5)
    val (trnSetNb, vldSetNb, tstSetNb) = (trainingSet.count.toDouble, validationSet.count.toDouble, testSet.count.toDouble)
    println("Distribution = " +(trnSetNb / M, vldSetNb / M, tstSetNb / M))                    // 0.6 : 0.2 : 0.2

    // Initialization                                                         `
    var w = initWeight

    // hypothesis
    def hypo(x: Vector) = {
      if (1 / (1 + exp(-w dot x)) > testThreshold) 1 else 0
    }

    // cost function
    def costFunction = {
      val negativeInfinity = -1E10
      def logLike(hypoRes: Int) = if (hypoRes == 1) 0.0 else negativeInfinity
      testSet.map(pt => pt.y * logLike(hypo(pt.x)) + (1 - pt.y) * logLike(1 - hypo(pt.x))).reduce(_ + _) / -M
    }

    // Training
    val costList = for (i <- 1 until iteration) yield {
      val gradient = trainingSet.map {
        p => (1 / (1 + exp(-w dot p.x)) - p.y) * p.x
      }.reduce(_ + _) / M
      w -= gradient * learningRate
      val cost = costFunction
      println("cost_" + i + " = " + cost)
      (i, cost)
    }

    // Cross Validation
    // TODO

    // Test
    val testRes = testSet.map {
      p => (hypo(p.x) == 1 && p.y == 1) || (hypo(p.x) == 0 && p.y == 0)
    }
    val genErr = testRes.countByValue.apply(false) / tstSetNb
    println("Generalization Error = " + genErr)
    plotLogRegFigures

    // Visualisation
    def plotLogRegFigures() = {

      val (iterSeq, costSeq) = costList.unzip
      learningCurvePlot(iterSeq.toArray, costSeq.toArray)

      /**
       * Note that boundary visualization is a 2dPlot
       * PCA is needed for projecting data to the first plane
       */

      subplot(2, 1, 2)

      val positive = scaledPoints.filter(_.y == 1)
      val negative = scaledPoints.filter(_.y == 0)
      val pos_x = positive.map(_.x(0)).collect
      val pos_y = positive.map(_.x(1)).collect
      val neg_x = negative.map(_.x(0)).collect
      val neg_y = negative.map(_.x(1)).collect
      val seq = DenseVector.tabulate(1000)(x => (x / 1000.0) * 4 - 2)

      plot.hold = true
      plot(pos_x, pos_y, '.')
      plot(neg_x, neg_y, '.')
      plot(seq, -seq * w(1) / w(2) - w(0) / w(2))
      xlabel("x1")
      ylabel("x2")
    }
  }

  def main(args: Array[String]) {

    val w = Vector(D + 1, _ => 1) // data generalized =>  D + 1
    val iteration = 50
    val learningRate = 1
    val testThreshold = 0.5
    localLogReg(w, iteration, learningRate, testThreshold)
  }
}