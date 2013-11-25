package SVM_with_soft_margin

import scala.io.Source
import breeze.linalg.DenseVector
import scala.math.pow
import libsvm._
import libsvm.svm._
import scala._
import scala.util.Random

/**
 * Created with IntelliJ IDEA.
 * User: coderh
 * Date: 11/25/13
 * Time: 10:54 AM
 */
object kernelMethod {

  case class DataPoint(x: DenseVector[Double], y: Double) {
    def this(arr: Array[Double]) {
      this(DenseVector(arr.tail), arr.head)
    }

    def errorMeasure(weight: DenseVector[Double]) = {
      if (this.x.dot(weight) * this.y > 0) 0 else 1
    }
  }

  def extendPoint(pt: DataPoint) = {
    val x = DenseVector(1.0 +: pt.x.toArray)
    DataPoint(x, pt.y)
  }

  def loadData(path: String) = {
    val data = for (line <- Source.fromFile(path).getLines()) yield
      new DataPoint(line.replaceAll("^\\s+", "").split("\\s+").map(_.toDouble))
    data.toList
  }

  def XvsAll(pt: DataPoint, x: Double) = {
    val label = if (pt.y == x) 1 else -1
    DataPoint(pt.x, label)
  }

  def kernel(xn: DenseVector[Double])(xm: DenseVector[Double]) = {
    pow(1 + xn.dot(xm), 2)
  }

  def vector2svmNode(v: DenseVector[Double]) = Array(new svm_node() {
    this.index = 1
    this.value = v(0)
  }, new svm_node() {
    this.index = 2
    this.value = v(1)
  })

  def learning(trainingSet: Partition, testSet: Partition, cost: Double, Q: Int) = {
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

    val param1 = new svm_parameter {
      this.svm_type = svm_parameter.C_SVC
      this.kernel_type = svm_parameter.POLY
      this.gamma = 1
      this.coef0 = 1
      this.degree = Q
      this.eps = 0.001
      this.nr_weight = 0
      this.cache_size = 10
      this.C = cost
    }

    //    val param2 = new svm_parameter {
    //      this.svm_type = svm_parameter.C_SVC
    //      this.kernel_type = svm_parameter.RBF
    //      this.gamma = 1
    //      this.eps = 0.001
    //      this.nr_weight = 0
    //      this.cache_size = 10
    //      this.C = cost
    //    }

    val param = param1

    //disable display
    val display = new svm_print_interface() {
      def print(s: String) {
      }
    }
    svm.svm_set_print_string_function(display)

    val check = svm.svm_check_parameter(prob, param)
    val model = if (check == null) svm.svm_train(prob, param) else throw new Error("Wrong SVM parameters: " + check)
    val E_in = trainingSet.count(pt => svm_predict(model, vector2svmNode(pt.x)) * pt.y < 0).toDouble / trainingSet.size
    val E_out = testSet.count(pt => svm_predict(model, vector2svmNode(pt.x)) * pt.y < 0).toDouble / testSet.size
    val SVs = model.SV.toList.map {
      case xs: Array[svm_node] => DenseVector[Double](xs(0).value, xs(1).value)
    }
    (E_in, E_out, SVs.size)

    //    val coefficients = model.sv_coef.toList.head.toList
    //    val refPoint = trainingSet.find(_.x == SVs.head) match {
    //      case Some(x) => x
    //      case None => throw new Error("SV is not in training set")
    //    }

    //    val b = refPoint.y - (SVs.map(pts => kernel(pts)(refPoint.x)) zip coefficients map (p => p._1 * p._2)).sum
    //    val b_bis = model.rho.head

    //    val hypo = (x: DenseVector[Double]) => (SVs.map(sv => kernel(sv)(x)) zip coefficients map (p => p._1 * p._2)).sum + b_bis

    //    In - Sample - Error
    //    val E_in = trainingSet.count(pt => hypo(pt.x) * pt.y < 0).toDouble / trainingSet.size
  }

  val nbFold = 10
  type Partition = List[DataPoint]

  def crossValidation(folds: List[Partition], cost: Double, Q: Int) = {
    require(folds.size == nbFold)
    val indexedPartition = folds.zipWithIndex
    (0 to (nbFold - 1) map {
      i =>
        val validationSet = folds(i)
        val trainingSet = indexedPartition.filter(_._2 != i).map(_._1).reduce(_ ::: _)
        learning(trainingSet, validationSet, cost, Q)._2
    }).sum / nbFold
  }

  def run() = {

    //    val trainingSet = loadData("features.train.txt").map(pt => XvsAll(pt, label))
    //    val testSet = loadData("features.test.txt").map(pt => XvsAll(pt, label))

    val trainingSet = loadData("features.train.txt").filter(pt => pt.y == 1.00 || pt.y == 5.00).map(pt => XvsAll(pt, 1))
    //    val testSet = loadData("features.test.txt").filter(pt => pt.y == 1.00 || pt.y == 5.00).map(pt => XvsAll(pt, 1))

    val nbElem = trainingSet.size / nbFold
    val shuffledDataPoints = Random.shuffle(trainingSet)

    def partitionDataSet(set: List[DataPoint]): List[List[DataPoint]] = {
      if (set.size > nbElem) set.take(nbElem) :: partitionDataSet(set.drop(nbElem)) else Nil
    }

    val partitions: List[Partition] = partitionDataSet(shuffledDataPoints)

//    val costList = List(0.0001, 0.001, 0.01, 0.1, 1.0)
//    val pair = costList.map(c => (c, crossValidation(partitions, c, 2)))
//    val minErr = pair.map(_._2).min
//    val C = pair.filter(_._2 == minErr).map(_._1).min
//    C

    crossValidation(partitions, 0.001, 2)

    //    List(0.01, 1d, 100d, 10000d, 1000000d).map(c => (c, learning(trainingSet, testSet, c, 0)._2))
  }


  def main(args: Array[String]) = {
    //    val res1 = List[Double](0.0001, 0.001, 0.01, 1).map(c => run(c, 2))

    //    println(res1.mkString("\n"))

        println((1 to 100 map (i => run())).toList.sum / 100)


//    val lst = (1 to 100 map (i => run())).toList.groupBy(x => x).map(p => (p._1, p._2.size))
//    println(lst)
  }

}
