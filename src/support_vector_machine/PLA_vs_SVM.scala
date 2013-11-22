package support_vector_machine

import scala.util.Random
import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.plot._

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
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

  def plotData(dataSet: List[DataPoint], tf: targetFunc) = {
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
    p += plot(xs, xs.map(x => (tf.weights(0) + tf.weights(1) * x) / -tf.weights(2)), '-', "red")

    // draw g(x)
    //    p += plot(xs, xs.map(x => (w(0) + w(1) * x) / -w(2)), '-', "blue")

    // draw data
    val oneSide = dataSet.filter(_.y > 0).map(_.x)
    val theOtherSide = dataSet.filter(_.y < 0).map(_.x)
    p += plot(oneSide.map(_(0)), oneSide.map(_(1)), '+')
    p += plot(theOtherSide.map(_(0)), theOtherSide.map(_(1)), '.')
  }

}

object PLA_vs_SVM extends experiment {

  // TODO: correct it!
  // dimension
  val d = 2

  // nb of data points
  val N = 10

  def main(args: Array[String]) = {

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

    SVM.run(dataSet, tf)
    //    PLA.run(dataSet, tf)

    //    val iter = 1000
    //    val perf_pla = for (i <- 1 to iter) yield PLA.run()
    //    val perf_svm = for (i <- 1 to iter) yield SVM.run_spark()
    //    val percentage = (perf_pla zip perf_svm).count(p => p._1 > p._2).toDouble / iter
    //    println(percentage)
  }


  //  val sc = new SparkContext("local[2]", "SparkLR", System.getenv("SPARK_HOME"), null)

  object SVM {

    def run(dataSet: List[DataPoint], tf: targetFunc) = {

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
        this.eps = 0.0001
        this.nr_weight = 0
        this.shrinking = 0
        this.probability = 0
      }

      plotData(dataSet, tf)

      val model: svm_model = svm.svm_train(prob, param)

      //      val one = DenseVector.tabulate(N)(_ => 1.0)
      //      val Q = DenseMatrix.tabulate(N, N)((i: Int, j: Int) => (dataSet(i).x dot dataSet(j).x) * dataSet(i).y * dataSet(j).y)
      //      val alpha = inv(Q) * one
      val res = prob.x(0)(1).value
      println(model.l)
    }
  }
  object PLA {

    def run(dataSet: List[DataPoint], tf: targetFunc) = {

      var w = DenseVector[Double](0, 0, 0)
      val extendedDataSet = dataSet.map(extendPoint)

      def isMisclassified(pt: DataPoint) = (pt.x dot w) * pt.y <= 0

      var misclassifiedSet = extendedDataSet.filter(isMisclassified)
      while (!misclassifiedSet.isEmpty) {
        val missedPt = Random.shuffle(misclassifiedSet.toList).head
        w = w + missedPt.x * missedPt.y
        misclassifiedSet = extendedDataSet.filter(isMisclassified)
      }

      plotData(dataSet, tf)

      // error
      val n = 100000
      //    val err = sc.parallelize(Array.tabulate(n)(_ => generatePoint), 2).cache().filter(isMisclassified).count.toDouble / n
      val err = List.tabulate(n)(_ => generateDataPoint(d, tf)).map(extendPoint).count(isMisclassified).toDouble / n

      err
      //    println(w)
      //    println(err)
    }
  }
}