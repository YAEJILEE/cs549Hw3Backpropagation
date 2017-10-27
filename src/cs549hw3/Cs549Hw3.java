/*
 * Kevin Leehan
 * CS549
 * HW3
 * Backpropagation Algorithm
 *
 * I have set my epoch loop to end at 5million.  
 * I was unable to get the result using the architecture 2-2-1
 *
 */


package cs549hw3;

public class Cs549Hw3 {
             
    public static void main(String[] args) {
        //Name and class information for the output page
        System.out.println("Kevin Leehan\nCS549\nHW3\nBackpropagation Model\n");
        Backpropagation a = new Backpropagation();
        Backpropagation b = new Backpropagation();
        Backpropagation c = new Backpropagation();
        Backpropagation d = new Backpropagation();
        
        //2-hidden 1-output
        a.architecture(2,1);
        System.out.println();
        
        //3-hidden 1-output
        b.architecture(3,1);
        System.out.println();
        
        //4-hidden 2-output
        c.architecture(4,2);
        System.out.println();
        
        //6-hidden 1-output
        d.architecture(6,1);
        
        
       
    } //End main
      
}