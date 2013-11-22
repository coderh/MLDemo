package support_vector_machine

import scala.util.Random
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot._
import libsvm._

/**
 *
 * Author: coderh
 * Date: 11/17/13
 * Time: 6:08 PM
 *
 */

trait experiment {

  case class targetFunc(weights: List[Double]) {
    require(weights.size == 3)

    def apply(x: DenseVector[Double]) = if ((DenseVector(weights(0), weights(1), weights(2)) dot x) > 0) 1.0 else -1.0
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

  def generateDataPoint(dim: Int, tFunc: targetFunc) = {
    val x = DenseVector.tabulate(dim)(_ => 2 * Random.nextDouble() - 1)
    DataPoint(x, tFunc(DenseVector(1.0 +: x.toArray)))
  }

}

object PLA_vs_SVM extends experiment {

  // dimension
  val d = 2

  // nb of data points
  val N = 10

  def plotData(dataSet: List[DataPoint],
               tf: targetFunc,
               plaWeights: List[Double] = Nil,
               svmWeights: List[Double] = Nil,
               sptVcts: List[DenseVector[Double]] = Nil) = {

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
    p += plot(xs, xs.map(x => (tf.weights(0) + tf.weights(1) * x) / -tf.weights(2)), '-', "r")

    // draw data
    val oneSide = dataSet.filter(_.y > 0).map(_.x)
    val theOtherSide = dataSet.filter(_.y < 0).map(_.x)
    p += plot(oneSide.map(_(0)), oneSide.map(_(1)), '.', "g")
    p += plot(theOtherSide.map(_(0)), theOtherSide.map(_(1)), '.', "m")

    // draw hypo
    if (!plaWeights.isEmpty)
      p += plot(xs, xs.map(x => (plaWeights(0) + plaWeights(1) * x) / -plaWeights(2)), '-', "b")

    if (!sptVcts.isEmpty && !svmWeights.isEmpty) {
      p += plot(sptVcts.map(_(0)), sptVcts.map(_(1)), '+')
      p += plot(xs, xs.map(x => (svmWeights(0) + svmWeights(1) * x) / -svmWeights(2)), '-', "b")
    }

    p
  }

  def singleRun() = {
    // settings
    val pt1 = DenseVector(2 * Random.nextDouble() - 1, 2 * Random.nextDouble() - 1)
    val pt2 = DenseVector(2 * Random.nextDouble() - 1, 2 * Random.nextDouble() - 1)
    val ws = List(pt2(1) * pt1(0) - pt1(1) * pt2(0), pt1(1) - pt2(1), pt2(0) - pt1(0))
    val tf = targetFunc(ws)

    // generate data sets
    var dataSet = List.tabulate(N)(_ => generateDataPoint(d, tf))
    while (dataSet.forall(_.y > 0) || dataSet.forall(_.y < 0)) {
      dataSet = List.tabulate(N)(_ => generateDataPoint(d, tf))
    }

    SVM.run(dataSet, tf, true)
    //(SVM.run(dataSet, tf), PLA.run(dataSet, tf))

  }

  def main(args: Array[String]) = {

    singleRun

//    val iteration = 1000
//    val percentage = (1 to iteration).map(x => singleRun()).count(p => p._1 < p._2).toDouble / iteration
//
//    println
//    println(percentage)
  }

  object SVM {

    def solver(dataSet: List[DataPoint], tf: targetFunc) = {
      val one = DenseVector.tabulate(N)(_ => 1.0)
      val Q = DenseMatrix.tabulate(N, N)((i: Int, j: Int) => (dataSet(i).x dot dataSet(j).x) * dataSet(i).y * dataSet(j).y)
      (Q, one)
    }

    def run(dataSet: List[DataPoint], tf: targetFunc, draw: Boolean = false) = {

      val prob = new svm_problem() {
        this.l = dataSet.size
        this.y = dataSet.map(_.y).toArray
        this.x = dataSet.map(_.x).map {
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
        this.svm_type = 0
        this.kernel_type = 0
        this.eps = 0.001
        this.nr_weight = 0
        this.cache_size = 10
        this.C = Double.PositiveInfinity
      }

      val check = svm.svm_check_parameter(prob, param)
      val model = if (check == null) svm.svm_train(prob, param) else throw new Error("Wrong SVM parameters: " + check)
      val svs = model.SV.toList.map {
        case xs: Array[svm_node] => DenseVector[Double](xs(0).value, xs(1).value)
      }

      val coefficients = model.sv_coef.toList.head.toList
      val w = (svs zip coefficients map (p => p._1 * p._2)).reduce[DenseVector[Double]](_ + _)
      val b = tf(DenseVector(1.0 +: svs.head.toArray)) - w.dot(svs.head)
      val hypoWeights = DenseVector(b +: w.toArray)

      println(hypoWeights)

      if (draw)
        plotData(dataSet, tf, svmWeights = hypoWeights.toArray.toList, sptVcts = svs)

      def isMisclassified(pt: DataPoint) = (pt.x dot hypoWeights) * pt.y <= 0

      // error
      val n = 100000
      val err = List.tabulate(n)(_ => generateDataPoint(d, tf)).map(extendPoint).count(isMisclassified).toDouble / n
      err
    }
  }

  object PLA {

    def run(dataSet: List[DataPoint], tf: targetFunc, draw: Boolean = false) = {

      var w = DenseVector[Double](0, 0, 0)
      val extendedDataSet = dataSet.map(extendPoint)

      def isMisclassified(pt: DataPoint) = (pt.x dot w) * pt.y <= 0

      var misclassifiedSet = extendedDataSet.filter(isMisclassified)
      while (!misclassifiedSet.isEmpty) {
        val missedPt = Random.shuffle(misclassifiedSet.toList).head
        w = w + missedPt.x * missedPt.y
        misclassifiedSet = extendedDataSet.filter(isMisclassified)
      }

      if (draw)
        plotData(dataSet, tf, plaWeights = w.toArray.toList)

      // error
      val n = 100000
      val err = List.tabulate(n)(_ => generateDataPoint(d, tf)).map(extendPoint).count(isMisclassified).toDouble / n
      err
    }
  }

}