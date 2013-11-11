package logistic_regression

import breeze.linalg._
import scala.math._
import scala.util.Random

/**
 * Created with IntelliJ IDEA.
 * User: coderh
 * Date: 11/4/13
 * Time: 10:11 AM
 */
object logReg {

  case class DataPoint(x: DenseVector[Double], y: Double)

  def g(s: Double) = 1 / (1 + exp(-s))

  def extendPoint(pt: DataPoint) = {
    val x = DenseVector(1.0 +: pt.x.toArray)
    DataPoint(x, pt.y)
  }

  val training_N = 100
  val test_N = 1000000
  val eta = 0.01
  val runNb = 100

  def run: Double = {

    val intercept = 2 * Random.nextDouble - 1

    def tgtFunc(pt: DenseVector[Double]) = {
      require(pt.length == 2)
      if (pt(0) + pt(1) + intercept >= 0) 1 else 0
    }

    def generatePoint = {
      val x = DenseVector(2 * Random.nextDouble - 1, 2 * Random.nextDouble - 1)
      val y = if (tgtFunc(x) == 1) 1 else -1
      DataPoint(x, y)
    }

    val dataSet = List.tabulate(training_N)(_ => generatePoint)
    val extendedDateSet = dataSet.map(extendPoint)

    var w, w_old = DenseVector[Double](0.0, 0.0, 0.0)

    def SGD(pt: DataPoint) = {
      val gradient = -pt.x * (pt.y / (1.0 + exp(pt.y * w.dot(pt.x))))
      w = w - gradient * eta
    }

    var cnt = 0
    do {
      // in one epoch
      w_old = w
      Random.shuffle(extendedDateSet).foreach(SGD)
      cnt += 1
    } while ((w_old - w).norm(2) >= 0.01)

    val Eout = List.tabulate(test_N)(_ => generatePoint).map(extendPoint).map(pt => {
      log(1.0 + exp(-pt.y * w.dot(pt.x)))
    }).sum / test_N

    //    cnt
    Eout
  }

  def main(args: Array[String]) = {
    val Eout_bar = (1 to runNb).map(_ => run).sum / runNb
    println(Eout_bar)
  }
}
