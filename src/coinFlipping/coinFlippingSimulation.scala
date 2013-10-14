package coinFlipping

import scala.util.Random

/**
 *
 * Author: coderh
 * Date: 10/10/13
 * Time: 10:11 PM
 *
 */
object coinFlippingSimulation {

  val repeat = 10
  val singleRunIteration = 1000
  val experimentIteration = 100000

  def flip = Random.nextBoolean()

  def trial = (1 to repeat).map(x => flip).count(_ == true).toDouble / 10

  def experiment() = {
    val sample = 0 until singleRunIteration map (i => (i, trial))
    (sample(0)._2, sample(Random.nextInt(singleRunIteration))._2, sample.map(_._2).min)
  }

  def addTriple(t1: (Double, Double, Double), t2: (Double, Double, Double)) = {
    (t1._1 + t2._1, t1._2 + t2._2, t1._3 + t2._3)
  }

  def main(args: Array[String]) {
    //    println(experiment)
    val res = (1 to experimentIteration).map(_ => experiment()).reduce(addTriple)
    println((res._1 / experimentIteration.toDouble, res._2 / experimentIteration.toDouble, res._3 / experimentIteration.toDouble))
  }
}
