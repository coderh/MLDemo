package perceptronLearning

import breeze.plot._
import org.apache.spark.util.Vector
import org.apache.spark.SparkContext
import scala.util.Random

/**
 * Created with IntelliJ IDEA.
 * User: cloudera
 * Date: 10/3/13
 * Time: 2:15 PM
 */

object PLA {

  val sc = new SparkContext("local[2]", "SparkLR", System.getenv("SPARK_HOME"), null)

  case class targetFunction(pt1: Vector, pt2: Vector) {
    val w2 = 1.0
    val w1 = -(pt1(2) - pt2(2)) / (pt1(1) - pt2(1))
    val w0 = -(pt1(2) * pt2(1) - pt2(2) * pt1(1)) / (pt2(1) - pt1(1))
    val tgw = Vector(w0, w1, w2)

    def apply(pt: Vector) = {
      tgw dot pt
    }
  }

  def generatePoint = {
    Vector(1, 2 * Random.nextFloat - 1, 2 * Random.nextFloat - 1)
  }

  def sign(x: Double) = {
    if (x > 0) 1 else -1
  }

  def runPLA(plotFlag: Boolean = false) = {

    val N = 100
    val tf = new targetFunction(generatePoint, generatePoint)
    val dataSet = Array.tabulate(N)(_ => generatePoint)

    var w = Vector(0, 0, 0)
    def hypo(v: Vector) = sign(w dot v)
    def isMisclassified(v: Vector) = hypo(v) * sign(tf(v)) < 0

    var cnt = 0
    var misclassifiedSet = for (x <- dataSet if isMisclassified(x)) yield x
    while (!misclassifiedSet.isEmpty) {
      val misclassifiedPoint = Random.shuffle(misclassifiedSet.toList).head
      w = w + misclassifiedPoint * sign(tf.apply(misclassifiedPoint))
      misclassifiedSet = for (x <- dataSet if isMisclassified(x)) yield x
      cnt += 1
    }

    // visualizations

    if (plotFlag) {

      val f = Figure()
      val p = f.subplot(0)

      // figure settings
      p.xlim = (-1, 1)
      p.ylim = (-1, 1)
      p.xlabel = "x"
      p.ylabel = "y"

      // two extremities of domain
      val xs = Array(-1.0, 1.0)

      // draw f(x)
      p += plot(xs, xs.map(-tf.w1 * _ + -tf.w0), '-', "red")

      // draw g(x)
      p += plot(xs, xs.map(x => (w(0) + w(1) * x) / -w(2)), '-', "blue")

      // draw data
      val oneSide = dataSet.filter(pt => sign(tf(pt)) == 1)
      val theOtherSide = dataSet.filter(pt => sign(tf(pt)) == -1)
      p += plot(oneSide.map(_(1)), oneSide.map(_(2)), '+')
      p += plot(theOtherSide.map(_(1)), theOtherSide.map(_(2)), '.')
    }

    // error
    val n = 100000
    val err = sc.parallelize(Array.tabulate(n)(_ => generatePoint), 2).cache().filter(isMisclassified).count.toDouble / n
    (cnt, err)
  }

  def main(args: Array[String]) {
    // single run
    runPLA(plotFlag = true)

    // multi-run
    val nbIteration = 50
    val p = for (i <- 1 until nbIteration) yield (i, runPLA())
    println("Iteration number = " + p.map(_._2._1).sum / nbIteration)
    println("Error rate = " + p.map(_._2._2).sum / nbIteration)
  }
}