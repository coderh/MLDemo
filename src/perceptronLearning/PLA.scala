package perceptronLearning

import scala.util.Random
import spark.util.Vector
import scalala.library.Plotting._

/**
 * Created with IntelliJ IDEA.
 * User: cloudera
 * Date: 10/3/13
 * Time: 2:15 PM
 */

object PLA {

  class targetFunction(pt1: Vector, pt2: Vector) {

    val w2 = 1
    val w1 = -(pt1(2) - pt2(2)) / (pt1(1) - pt2(1))
    val w0 = -(pt1(2) * pt2(1) - pt2(2) * pt1(1)) / (pt2(1) - pt1(1))
    val tgw = Vector(w0, w1, w2)

    def apply(pt: Vector) = {
      tgw dot pt
    }

    def plotBoundary(): Unit = {
      val xs = Array(-1, 1)
      plot.hold = true
      plot(xs, xs.map(-w1 * _ + -w0), '-', "blue")
      xlim(-1, 1)
      ylim(-1, 1)
      xlabel("x")
      ylabel("y")
    }
  }

  def runPLA(plotFlag: Boolean) = {
    val N = 10
    val tf = new targetFunction(generatePoint, generatePoint)
    val dataSet = Array.tabulate(N)(_ => generatePoint)
    var w = Vector(0, 0, 0)

    var cnt = 0
    var misclassifiedSet = for (x <- dataSet if sign(w dot x) != sign(tf(x))) yield x
    while (!misclassifiedSet.isEmpty) {
      val misclassifiedPoint = Random.shuffle(misclassifiedSet.toList).head
      w = w + misclassifiedPoint * sign(tf.apply(misclassifiedPoint))
      misclassifiedSet = for (x <- dataSet if (w dot x) * tf.apply(x) < 0) yield x
      cnt += 1
    }

    if (plotFlag) {
      val xs = Array(-1, 1)
      tf.plotBoundary()
      plot.hold = true
      plot(xs, xs.map(x => (w(0) + w(1) * x) / -w(2)), '-', "red")
      plot(dataSet.filter(pt => sign(tf(pt)) == 1).map(_(1)), dataSet.filter(pt => sign(tf(pt)) == 1).map(_(2)), '+')
      plot(dataSet.filter(pt => sign(tf(pt)) == -1).map(_(1)), dataSet.filter(pt => sign(tf(pt)) == -1).map(_(2)), '.')

      xlim(-1, 1)
      ylim(-1, 1)
      xlabel("x")
      ylabel("y")
    }

    cnt
  }

  def generatePoint = {
    Vector(1, 2 * Random.nextFloat - 1, 2 * Random.nextFloat - 1)
  }

  def sign(x: Double) = {
    if (x > 0) 1 else -1
  }

  def main(args: Array[String]) {
    runPLA(plotFlag = true)
    //    val p = for (i <- 1 until 1000) yield (i, runPLA(plotFlag = false))
    //    println(p.map(_._2).sum / 1000)
  }
}
