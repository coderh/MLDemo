package linear_regression

import org.apache.spark.util.Vector
import scala.util.Random
import breeze.plot._
import breeze.linalg._

/**
 *
 * Author: coderh
 * Date: 10/12/13
 * Time: 12:42 AM
 *
 */

object linReg {

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
    if (x > 0) 1.0 else -1.0
  }

  def run(plotFlag: Boolean = false) = {

    val N = 10
    val D = 3
    val tf = new targetFunction(generatePoint, generatePoint)

    val dataSet = Array.tabulate(N)(_ => generatePoint)
    val x = DenseMatrix.tabulate(N, D)((i: Int, j: Int) => dataSet(i)(j))
    val y = DenseVector(dataSet.map(x => sign(tf(x))))
    val xt = x.t
    val weight = inv(xt * x) * xt * y
    var w = weight

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


    def hypo(v: Vector) = sign(w dot DenseVector(v.elements))
    def isMisclassified(v: Vector) = hypo(v) * sign(tf(v)) < 0

    var cnt = 0
    var misclassifiedSet = for (x <- dataSet if isMisclassified(x)) yield x
    while (!misclassifiedSet.isEmpty) {
      val misclassifiedPoint = Random.shuffle(misclassifiedSet.toList).head
      w = w + DenseVector(misclassifiedPoint.elements) * sign(tf.apply(misclassifiedPoint))
      misclassifiedSet = for (x <- dataSet if isMisclassified(x)) yield x
      cnt += 1
    }

    // Ein
    //    val err_in = dataSet.count(isMisclassified).toDouble / N
    //
    //        // Eout
    //        val n = 1000
    //        val err_out = List.tabulate(n)(_ => generatePoint).count(isMisclassified).toDouble / n
    //        (err_in, err_out)
    cnt
  }

  def main(args: Array[String]) {

    // nb of iteration
    val iteration = 1000

    //        val (in, out) = (1 to iteration).map(_ => runPLA()).unzip
    //        println("result: " +(in.sum / iteration, out.sum / iteration))

    val res = (1 to iteration).map(_ => run()).sum / iteration
    println(res)
  }
}