package support_vector_machine

import breeze.linalg.DenseVector
import breeze.plot._
import libsvm._
import scala.Array
import libsvm.svm._

/**
 * Created with IntelliJ IDEA.
 * User: coderh
 * Date: 12/3/13
 * Time: 9:56 AM
 */
object SVM {

  case class DataPoint(x: DenseVector[Double], y: Double) {
    def this(arr: Array[Double]) {
      this(DenseVector(arr.tail), arr.head)
    }

    def errorMeasure(weight: DenseVector[Double]) = {
      if (this.x.dot(weight) * this.y > 0) 0 else 1
    }
  }

  def plotData(dataSet: List[DataPoint],
               sptVcts: List[DenseVector[Double]] = Nil) = {

    val f = Figure()
    val p = f.subplot(0)

    // figure settings
    p.xlim = (-6, 6)
    p.ylim = (-6, 6)
    p.xlabel = "x"
    p.ylabel = "y"

    // two extremities of domain
    val xs = Array(-1.0, 1.0)


    // draw data
    val oneSide = dataSet.filter(_.y > 0).map(_.x)
    val theOtherSide = dataSet.filter(_.y < 0).map(_.x)
    p += plot(oneSide.map(_(0)), oneSide.map(_(1)), '.', "g")
    p += plot(theOtherSide.map(_(0)), theOtherSide.map(_(1)), '.', "m")

    // draw hypo

    if (!sptVcts.isEmpty) {
      p += plot(sptVcts.map(_(0)), sptVcts.map(_(1)), '+')
    }
  }

  def nonLinearTrans(point: DataPoint) = {
    require(point.x.length == 2)
    val x1 = point.x(0)
    val x2 = point.x(1)
    DataPoint(DenseVector(x2 * x2 - 2 * x1 - 1, x1 * x1 - 2 * x2 + 1), point.y)
  }

  def learning(trainingSet: List[DataPoint]) = {
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
      this.kernel_type = svm_parameter.POLY
      this.gamma = 1
      this.coef0 = 1
      this.degree = 2
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

    SVs

  }

  def run() = {
    val trainingSet = List(
      DataPoint(DenseVector[Double](1, 0), -1),
      DataPoint(DenseVector[Double](0, 1), -1),
      DataPoint(DenseVector[Double](0, -1), -1),
      DataPoint(DenseVector[Double](-1, 0), 1),
      DataPoint(DenseVector[Double](0, 2), 1),
      DataPoint(DenseVector[Double](0, -2), 1),
      DataPoint(DenseVector[Double](-2, 0), 1)
    )


    plotData(trainingSet, learning(trainingSet))
//    println()
  }

  def main(args: Array[String]) = {
    run()
  }
}
