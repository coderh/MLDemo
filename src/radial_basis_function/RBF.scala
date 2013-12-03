package radial_basis_function

import breeze.linalg.DenseVector
import scala.util.Random
import scala.math._
import libsvm._
import scala.Array
import libsvm.svm._
import breeze.plot._

/**
 *
 * Author: coderh
 * Date: 12/3/13
 * Time: 9:40 PM
 *
 */


object RBF {

  def targetFunc(x: DenseVector[Double]): Double = {
    require(x.length == 2)
    val x1 = x(0)
    val x2 = x(1)
    if (x2 - x1 + 0.25 * sin(Pi * x1) > 0) 1 else -1
  }

  case class DataPoint(x: DenseVector[Double], y: Double) {
    def this(arr: Array[Double]) {
      this(DenseVector(arr.init), arr.last)
    }

    def errorMeasure(weight: DenseVector[Double]) = {
      if (this.x.dot(weight) * this.y > 0) 0 else 1
    }
  }

  def extendPoint(pt: DataPoint) = {
    val x = DenseVector(1.0 +: pt.x.toArray)
    DataPoint(x, pt.y)
  }

  def generateDataPoint(dim: Int) = {
    val x = DenseVector.tabulate(dim)(_ => 2 * Random.nextDouble() - 1)
    DataPoint(x, targetFunc(x))
  }

  def vector2svmNode(v: DenseVector[Double]) = Array(new svm_node() {
    this.index = 1
    this.value = v(0)
  }, new svm_node() {
    this.index = 2
    this.value = v(1)
  })

  def plotData(dataSet: List[DataPoint],
               SVs: List[DenseVector[Double]] = Nil) = {

    val f = Figure()
    val p = f.subplot(0)

    // figure settings
    p.xlim = (-1, 1)
    p.ylim = (-1, 1)
    p.xlabel = "x"
    p.ylabel = "y"

    // two extremities of domain
    //    val xs = Array(-1.0, 1.0)

    val xs: Array[Double] = Array.tabulate(1000)(x => -1.0 + 2.0 / 1000 * x)

    // draw f(x)
    p += plot(xs, xs.map(x => x - 0.25 * sin(Pi * x)), '-', "r")

    // draw data
    val oneSide = dataSet.filter(_.y > 0).map(_.x)
    val theOtherSide = dataSet.filter(_.y < 0).map(_.x)
    p += plot(oneSide.map(_(0)), oneSide.map(_(1)), '.', "g")
    p += plot(theOtherSide.map(_(0)), theOtherSide.map(_(1)), '.', "m")

    // draw hypo

    if (!SVs.isEmpty) {
      p += plot(SVs.map(_(0)), SVs.map(_(1)), '+')
    }
  }

  def kernel(SV: DenseVector[Double])(X: DenseVector[Double]) = {
    val gamma = 1.5
    exp(-gamma * (SV - X).dot(SV - X))
  }

  def RBF_kernel() = {
    val N = 100
    val D = 2
    val trainingSet = List.tabulate(N)(_ => generateDataPoint(D))

    val prob = new svm_problem() {
      this.l = trainingSet.size
      this.y = trainingSet.map(_.y).toArray
      this.x = trainingSet.map(_.x).map {
        case v: DenseVector[Double] =>
          Array(new svm_node() {
            this.index = 1
            this.value = v(0)
          }, new svm_node() {
            this.index = 2
            this.value = v(1)
          })
      }.toArray
    }

    val param = new svm_parameter {
      this.svm_type = svm_parameter.C_SVC
      this.kernel_type = svm_parameter.RBF
      this.gamma = 1.5
      this.eps = 0.001
      this.nr_weight = 0
      this.cache_size = 10
      this.C = Double.PositiveInfinity
    }

    //disable display
    val display = new svm_print_interface() {
      def print(s: String) {
      }
    }
    svm.svm_set_print_string_function(display)

    val check = svm.svm_check_parameter(prob, param)
    val model = if (check == null) svm.svm_train(prob, param) else throw new Error("Wrong SVM parameters: " + check)

    val SVs = model.SV.toList.map {
      case xs: Array[svm_node] => DenseVector[Double](xs(0).value, xs(1).value)
    }

    //    plotData(trainingSet, SVs)

    //    val E_in = trainingSet.count(pt => svm_predict(model, vector2svmNode(pt.x)) * pt.y < 0).toDouble / trainingSet.size
    //    val b = model.rho.head
    val coef = model.sv_coef.head.toList
    val refPoint = trainingSet.find(_.x == SVs.head) match {
      case Some(x) => x
      case None => throw new Error("SV is not in training set")
    }
    val b = refPoint.y - (coef zip SVs).map(p => p._1 * kernel(p._2)(refPoint.x)).sum

    def g(x: DenseVector[Double]) = {
      val r = (coef zip SVs).map(p => p._1 * kernel(p._2)(x)).sum + b
      if (r > 0) 1 else -1
    }

    val E_in = trainingSet.map(pt => pt.y * g(pt.x)).count(_ < 0).toDouble / trainingSet.size
    //    (b, bb)
    E_in
  }

  def solution() = {
    val nbIter = 10000
    val res = (1 to nbIter).map(_ => RBF_kernel).count(_ != 0.0).toDouble / nbIter.toDouble
    println(res)
  }

  def RBF_regular() = {

  }

  def main(args: Array[String]) = {
    solution
    //    println(SVM)
  }
}
