package nonlinearTransformation

import scala.util.Random
import breeze.linalg._
import breeze.plot._

/**
 *
 * Author: coderh
 * Date: 10/12/13
 * Time: 11:52 AM
 *
 */
object nonLinTrans {

  case class DataPoint(x: DenseVector[Double], y: Double)

  def sign(x: Double) = {
    if (x > 0) 1.0 else -1.0
  }

  def targetFunction(x: DenseVector[Double]) = {
    sign(x(1) * x(1) + x(2) * x(2) - 0.6)
  }

  def generatePoint = {
    val x = DenseVector(1.0, 2 * Random.nextDouble - 1, 2 * Random.nextDouble - 1)
    val y = targetFunction(x)
    DataPoint(x, y)
  }


  def plotDataSet(dataSet: List[DataPoint]) = {
    val f = Figure()
    val p = f.subplot(0)

    // figure settings
    p.xlim = (-1, 1)
    p.ylim = (-1, 1)
    p.xlabel = "x"
    p.ylabel = "y"


    // draw data
    val oneSide = dataSet.filter(pt => pt.y == 1)
    val theOtherSide = dataSet.filter(pt => pt.y == -1)
    p += plot(oneSide.map(_.x(1)), oneSide.map(_.x(2)), '+')
    p += plot(theOtherSide.map(_.x(1)), theOtherSide.map(_.x(2)), '.')
    p
  }

  def nonLinearTransformation(pt: DataPoint) = {
    val x = pt.x
    DataPoint(DenseVector(1, x(1), x(2), x(1) * x(2), x(1) * x(1), x(2) * x(2)), pt.y)
  }

  def run = {

    val N = 1000
    val D = 6
    val noise_factor = 0.1

    def generateDataSet = {
      val dataSet = Array.tabulate(N)(_ => generatePoint).map(nonLinearTransformation)
      val (part_1, part_2) = Random.shuffle(dataSet.toList).splitAt((noise_factor * N).toInt)
      val part_1_noise = part_1.map(pt => DataPoint(pt.x, -pt.y))
      part_1_noise ::: part_2
    }

    val dataSet_noise = generateDataSet
    //    val ps = plotDataSet(dataSet_noise)

    val X = DenseMatrix.tabulate(N, D)((i: Int, j: Int) => dataSet_noise(i).x(j))
    val Xt = X.t
    val y = DenseVector(dataSet_noise.map(_.y).toArray)
    val w = inv(Xt * X) * Xt * y

    val ws = DenseVector(-1.0, -0.05, 0.08, 0.13, 1.5, 1.5) //0.96
    //    val ws = DenseVector(-1.0, -0.05, 0.08, 0.13, 1.5, 15) //0.67
    //    val ws = DenseVector(-1.0, -0.05, 0.08, 0.13, 15, 1.5) //0.666
    //    val ws = DenseVector(-1.0, -1.5, 0.08, 0.13, 0.05, 0.05) // 0.653
    //    val ws = DenseVector(-1.0, -0.05, 0.08, 1.5, 0.15, 0.15) // 0.569

    // draw g(x)
    //    val xs = Array(-1.0, 1.0)
    //    ps += plot(xs, xs.map(x => (w(0) + w(1) * x) / -w(2)), '-', "blue")


    def isMisclassified(pt: DataPoint) = {
      def hypo(v: DenseVector[Double]) = sign(w dot v)
      hypo(pt.x) * pt.y < 0
    }

    def agreement(pt: DataPoint) = {
      def hypo1(v: DenseVector[Double]) = sign(w dot v)
      def hypo2(v: DenseVector[Double]) = sign(ws dot v)
      hypo1(pt.x) * hypo2(pt.x) > 0
    }

    //    Ein
    //    dataSet_noise.count(isMisclassified).toDouble / N

    // inter-hypo agreement
    //    dataSet_noise.count(agreement).toDouble / N


    // Eout
    val n = 1000
    val err_out = generateDataSet.count(isMisclassified).toDouble / n

    err_out

  }

  def main(args: Array[String]) {
    //    println(run)

    val res = (1 to 1000).map(_ => run).sum / 1000
    println(res)
  }
}
