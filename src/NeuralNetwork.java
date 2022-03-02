public class NeuralNetwork {

    /** contains all weight matrices. 
     * <p> index 0: input -> hidden1 
     * <p> index length-1: hidden -> output
     * */
    private Matrix[] weight;

    /** contains all bias matrices. 
     * <p> index 0: hidden 1
     * <p> index length-1: output
     * */
    private Matrix[] bias;

    /** <p>Determines how quickly the network trains. Must be between 0 and 1. The default value is 0.1
     * <p>Higher values = faster training but less precision<p>
     * <p>Lower values = slower training but better precision<p>
     */
    public float trainingCoef = 0.1f;

    private int numI;
    private int numH;
    private int numO;
    private int numHL;
    
    public NeuralNetwork(int numInputs, int numHidden, int numOutputs, int numHLayers) {

        //initialize variables
        this.numI = numInputs;
        this.numH = numHidden;
        this.numO = numOutputs;
        this.numHL = numHLayers;

        //initialize weights/biases arrays
        weight = new Matrix[numHLayers + 1];
        bias = new Matrix[numHLayers + 1];

        //initialize weight matrices
        weight[0] = new Matrix(numHidden, numInputs);
        for (int i = 1; i < numHLayers; i++) {
            weight[i] = new Matrix(numHidden, numHidden);
        }
        weight[numHLayers] = new Matrix(numOutputs, numHidden);

        //initialize bias matrices
        for (int i = 0; i < numHLayers; i++) {
            bias[i] = new Matrix(numH, 1);
        }
        bias[numHL] = new Matrix(numOutputs, 1);

        //populate the weights/biases matrices their initial values
        for (int i = 0; i < bias.length; i++) {
            bias[i].setAll(1);
            weight[i].randomize();
        }
    }

    private Matrix feedLayer(Matrix input, Matrix weight, Matrix bias) {
        Matrix output;

        // Weights * Inputs
        output = weight.multiply(input);

        // add bias
        output.add(bias);

        // sigmoid function
        output.setDataFromList(sigmoid(output.getDataAsList()));

        return output;
    }

    //used internally to get outputs of all levels in network so they can be used in the backpropegation algorithm
    public Matrix[] getAllOutputs(float[] in) {
        Matrix input = new Matrix(numI, 1, in);
        Matrix[] outputs = new Matrix[numHL + 1];

        outputs[0] = feedLayer(input, weight[0], bias[0]);
        for (int i = 1; i < numHL + 1; i++) {
            outputs[i] = feedLayer(outputs[i - 1], weight[i], bias[i]);
        }

        return outputs;
    }

    //Used by end user to receive neuralnet final outputs as an array
    public float[] getOutputs(float[] input) {
        return getAllOutputs(input)[numHL].getDataAsList();
    }

    /** adjust weights and biases*/
    private Matrix[] calcDeltaWeights(Matrix weight, Matrix bias, Matrix nextNodeError, Matrix nextNodeOutput, Matrix previousNodeOutput) {
        Matrix dWeight = new Matrix(nextNodeOutput.getRows(), 1);
        dWeight.setDataFromList(dSigmoid(nextNodeOutput.getDataAsList()));
        dWeight = dWeight.schurProd(nextNodeError);
        dWeight = dWeight.schurProd(trainingCoef);

        return new Matrix[]{dWeight, dWeight.multiply(previousNodeOutput.transpose())};
    }

    public void train(float[] inputs, float[] answer) {
        if (answer.length != numO) {
            throw new IllegalArgumentException("answer array has invalid length. Expected length: " + numO + ". Received: " + answer.length);
        }

        Matrix[] error = new Matrix[numHL + 1];
        Matrix[] outputs = getAllOutputs(inputs);

        //initialize error matrices
        for (int i = 0; i < error.length; i++) {
            error[i] = new Matrix(bias[i].getRows(), bias[i].getCols());
        }

        //calculate the error of the output and store in the last slot of the error array (Error = Answer - Guess)
        for(int i = 0; i < error[error.length - 1].getRows(); i++) {
            error[error.length - 1].data[i][0] = answer[i] - outputs[error.length - 1].data[i][0];
        }

        // calculate errors of every matrix using the output error propogated backwards
        for(int i = error.length - 2; i >= 0; i--) {
            error[i] = weight[i + 1].transpose().multiply(error[i + 1]);
        }

        //moves backwards through the NN calculating the change in weights and biases and then updating them
        for(int i = numHL; i >= 0; i--) {
            Matrix[] dWeights = calcDeltaWeights(weight[i], bias[i], error[i], outputs[i], i > 0 ? outputs[i - 1] : new Matrix(numI, 1, inputs));
            bias[i] = bias[i].add(dWeights[0]);
            weight[i] = weight[i].add(dWeights[1]);
        }
    }

    // applies sigmoid function to array of nums
    private float[] sigmoid(float[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = sigmoid(inputs[i]);
        }
       return inputs;
    }

    // applies the derivative of the sigmoid function to an array of numbers
    private float[] dSigmoid(float[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = dSigmoid(inputs[i]);
        }
       return inputs;
    }

    private float sigmoid(float x) {
        return (float)(1f/(1f + Math.pow(Math.E, -x)));
    }

    private float dSigmoid(float x) {
        return x * (1 - sigmoid(x));
    }
    
    public void logWeights() {
        for (Matrix matrix : weight) {
            System.out.println(matrix + "\n");
        }
    }

    public void logBiases() {
        for (Matrix matrix : bias) {
            System.out.println(matrix + "\n");
        }
    }

    public String toString() {

        String out = "INPUT:  ";

        //visualize input nodes
        for (int i = 0; i < numI; i++) {
            out = out + "[input  " + i + "] ";
        }
        out = out + "\n";

        //visualize hidden nodes
        for (int i = 0; i < numHL; i++) {
            out = out + "HIDDEN: ";
            for (int j = 0; j < numH; j++) {
                out = out + "[node   " + j + "] ";
            }
            out = out + "\n";
        }

        //visualize output nodes
        out = out + "OUTPUT: ";
        for (int i = 0; i < numO; i++) {
            out = out + "[output " + i + "] ";
        }

        

        // //display hidden Weights
        // for (int i = 0; i < weight.length; i++) {
        //     out = out + "HIDDEN " + i + ": ";
        //     float[] data = weight[i].round(3).getDataAsList();
        //     for (int j = 0; j < data.length; j++) {
        //         out = out + data[j] + " ";
        //     }
        //     out = out + "\n";
        // }

        return out;
    }
}
