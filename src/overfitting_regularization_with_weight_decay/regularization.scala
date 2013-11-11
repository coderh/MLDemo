package overfitting_regularization_with_weight_decay

import scala.io.Source

/**
 *
 * Author: coderh
 * Date: 11/11/13
 * Time: 10:53 AM
 *
 */
object regularization {



  def main(args: Array[String]) {
    for (line <- Source.fromFile("in.dta.txt").getLines())
      println(line.replaceAll("^\\s+", "").split("\\s+").mkString(";"))
  }
}
