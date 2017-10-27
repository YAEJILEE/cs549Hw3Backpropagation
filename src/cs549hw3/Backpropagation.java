/*
 * Kevin Leehan
 * CS549
 * HW3
 * Backpropagation Algorithm
 */
package cs549hw3;

import java.util.Arrays;

public class Backpropagation{
    double learningCo=0.2;
    int inputUnits = 2;
    int hiddenUnits;
    int outputUnits;
    double[] input1 = {0,0,1,1};
    double[] input2 = {0,1,0,1};
    double[] target = {0,1,1,0};
    double[] actual = new double[target.length];
    double[][] initialInputWeights, initialHiddenWeights, 
               inputWeights, hiddenWeights; 
    double[] outputOfInput, 
             netOfHidden, outputOfHidden, deltaOfHidden,
             netOfOutput, outputOfOutput, deltaOfOutput;

    public void architecture(int a, int b){
        hiddenUnits=a;
        outputUnits=b;
        initialInputWeights = new double[inputUnits][hiddenUnits];
        initialHiddenWeights = new double[hiddenUnits][outputUnits]; 
        inputWeights = new double[inputUnits][hiddenUnits];
        hiddenWeights = new double[hiddenUnits][outputUnits];

        outputOfInput = new double[inputUnits];

        netOfHidden = new double[hiddenUnits];
        outputOfHidden = new double[hiddenUnits];
        deltaOfHidden = new double[hiddenUnits];

        netOfOutput = new double[outputUnits];
        outputOfOutput= new double[outputUnits];
        deltaOfOutput= new double[outputUnits];
        setRandomInitialWeights();
        startCalc();
    }
    
    public void startCalc(){
        int epoch=0;
        while(!Arrays.equals(actual, target)&&epoch<5000000){
            epoch++;         
            //Loop set input Values per epoch
            for(int i=0; i<input1.length; i++){
                outputOfInput[0]=input1[i];
                outputOfInput[1]=input2[i];
                
                //Calculates Net of Hidden Layer
                calcNetOfHidden();
                
                //calculate Output of Hidden Layer
                outputOfHidden = calcOutputOf(outputOfHidden, netOfHidden);
                
                //Calculates Net of Output Layer
                calcNetOfOutput();
                
                //Calculates the Output of Output Layer
                outputOfOutput = calcOutputOf(outputOfOutput, netOfOutput); 
                for(int j=0; j<outputOfOutput.length; j++){
                    outputOfOutput[j]=Math.floor(outputOfOutput[j]*100)/100;
                    //System.out.println(outputOfOutput[j]);
                    if(outputOfOutput[j]<.1)
                        outputOfOutput[j]=0;
                    if(outputOfOutput[j]>.9)
                        outputOfOutput[j]=1;
                    actual[i]=outputOfOutput[j];
                }
                
                deltaOfOutput = calcDeltaOfOutput(deltaOfOutput,target[i],outputOfOutput);
                               
                deltaOfHidden = calcDeltaOfHidden(deltaOfHidden, deltaOfOutput, hiddenWeights, outputOfHidden); 
                
                //Update weights
                if(actual[i]!=target[i]){
                    updateInputWeights();  
                    updateHiddenWeights();
                } 
            }//End of input value loop
        }//End of while loop
        //Prints the final results
        finalResults(epoch);                
    } //End of calculation
    
    
    //Sets the intial weights
    public void setRandomInitialWeights(){
        int i,j,k;
        //Setting weights between input and hidden layers
        for(i=0; i<inputUnits; i++)
            for(j=0; j<hiddenUnits; j++){
                double rNumb = ((Math.random()* 5 + 1)-3)/10;
                double r = Math.pow(10, 2);
                initialInputWeights[i][j] = Math.round(rNumb*r)/r;
                inputWeights[i][j]=initialInputWeights[i][j];
                //System.out.println(initialInputWeights[i][j]+" iWeight"+i+j);
                
            }
        //Setting weights between hidden and output layers
        for(j=0; j<hiddenUnits; j++)
            for(k=0; k<outputUnits; k++){
                double rNumb = ((Math.random()* 5 + 1)-3)/10;
                double r = Math.pow(10, 2);
                initialHiddenWeights[j][k] = Math.round(rNumb*r)/r;
                hiddenWeights[j][k]=initialHiddenWeights[j][k];
            }        
    }
    //Calculate the Net of Hidden Layer
    public void calcNetOfHidden(){
        for(int j=0; j<netOfHidden.length; j++){
            netOfHidden[j]=0;
            for (int i=0; i<outputOfInput.length; i++)
                netOfHidden[j]= netOfHidden[j]+(outputOfInput[i]*inputWeights[i][j]);
        }
    }
    //Calculate the Net Of Output Layer
    public void calcNetOfOutput(){
        for(int j=0; j<netOfOutput.length; j++){
            netOfOutput[j]=0;
            for (int i=0; i<outputOfHidden.length; i++)
                netOfOutput[j]= netOfOutput[j]+(outputOfHidden[i]*hiddenWeights[i][j]);
        }
    }
    //Used to calculate Output of hidden layer or Output of the Output layer
    public double[] calcOutputOf(double[] outputOfLayer, double[] netOfLayer){
        for(int k=0; k<outputOfLayer.length; k++)
            outputOfLayer[k]=1/(1+ Math.exp(-(netOfLayer[k])));
        return outputOfLayer;
    }
    //Calculate the delta of Output
    public double[] calcDeltaOfOutput(double[] deltaOfOutput, double target, double[] outputOfOutput){
        //Loops through each Output Unit
        for(int k=0; k<outputOfOutput.length; k++)
            deltaOfOutput[k]=(target-outputOfOutput[k])*outputOfOutput[k]*(1-outputOfOutput[k]);
        return deltaOfOutput;
    }
    //Calculate the Delta of the Hidden Layer
    public double[] calcDeltaOfHidden(double[] deltaOfHidden, double[] deltaOfOutput, double[][] hiddenWeights, double[] outputOfHidden){
        //Loops through each HiddenUnit
        for(int j=0; j<deltaOfHidden.length; j++){
            deltaOfHidden[j]=0;
            //Loops through each output of the hidden unit[j] and sums the Deltas*weights
            for(int k=0; k<deltaOfOutput.length; k++)
                deltaOfHidden[j]=deltaOfHidden[j]+(deltaOfOutput[k]*hiddenWeights[j][k]);
            deltaOfHidden[j]=deltaOfHidden[j]*outputOfHidden[j]*(1-outputOfHidden[j]);
        }
        return deltaOfHidden;
    }
    public void updateInputWeights(){
    //inputWeights[0][0]+(learningCo*deltaOfInput[0]*input1[0])
        for(int i=0; i<outputOfInput.length; i++)
            for(int j=0; j<deltaOfHidden.length; j++)
                inputWeights[i][j] = inputWeights[i][j] + (learningCo * deltaOfHidden[j] * outputOfInput[i]);
    }
    public void updateHiddenWeights(){
        //inputWeights[0][0]+(learningCo*deltaOfHidden[0]*input1[0])
        for(int i=0; i<outputOfHidden.length; i++){
            //Loops through each output of the deltaLayer[j] and sums the Deltas*weights
            for(int j=0; j<deltaOfOutput.length; j++){
                hiddenWeights[i][j] = hiddenWeights[i][j] + (learningCo * deltaOfOutput[j] * outputOfHidden[i]);
                //System.out.println(hiddenWeights[i][j]); 
            }
        }
    } 
    //prints an array, used for testing
    public void print(double[] a){
        for(int i=0; i<a.length;i++)
            System.out.println(a[i]);
    }
    //Displays final results
    public void finalResults(int epoch){
        System.out.println("Architecture: 2-"+hiddenUnits+"-"+outputUnits);
        System.out.println("Final epoch: "+epoch);
        System.out.println("Actual Output:");
        print(actual);
        System.out.println("Initial Input Weights: "+Arrays.deepToString(initialInputWeights));
        System.out.println("Final Input Weights: "+Arrays.deepToString(inputWeights));
        System.out.println("Initial Hidden Weights: "+Arrays.deepToString(initialHiddenWeights));
        System.out.println("Final Hidden Weights: "+Arrays.deepToString(hiddenWeights));
        System.out.println("----------------------------------------------");
    }
}
