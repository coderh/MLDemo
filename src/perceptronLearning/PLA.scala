package perceptronLearning

import scala.util.Random
import org.apache.spark.util.Vector

import breeze.plot._

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

    def plotBoundary() = {
      val xs = Array(-1, 1)

      val f = Figure()
      val p = f.subplot(0)
      p += plot(xs.map(_.toDouble), xs.map(-w1 * _ + -w0), '-', "blue")
      p.xlim = (-1, 1)
      p.ylim = (-1, 1)
      p.xlabel = "x"
      p.ylabel = "y"
      p
    }
  }

  def runPLA(plotFlag: Boolean) = {
    val N = 100
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
      val xs = Array(-1.0, 1.0)
      val p = tf.plotBoundary()

      p += plot(xs, xs.map(x => (w(0) + w(1) * x) / -w(2)), '-', "red")
      val oneSide = dataSet.filter(pt => sign(tf(pt)) == 1)
      val theOtherSide = dataSet.filter(pt => sign(tf(pt)) == -1)
      p += plot(oneSide.map(_(1)), oneSide.map(_(2)), '+')
      p += plot(theOtherSide.map(_(1)), theOtherSide.map(_(2)), '.')
      p.xlim = (-1, 1)
      p.ylim = (-1, 1)
      p.xlabel = "x"
      p.ylabel = "y"
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
