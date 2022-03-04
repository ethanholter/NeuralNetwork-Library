import java.util.Arrays;

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

        // [insert helpful comment here]
        this.numI = numInputs;
        this.numH = numHidden;
        this.numO = numOutputs;
        this.numHL = numHLayers;

        //initialize weights/biases arrays
        weight = new Matrix[numHLayers + 1];
        bias = new Matrix[numHLayers + 1];

        //initialize weight matrix for input -> hidden layer  1
        weight[0] = new Matrix(numHidden, numInputs);

        //initialize weights from hidden layer to hidden layer
        for (int i = 1; i < numHLayers; i++) {
            weight[i] = new Matrix(numHidden, numHidden);
        }

        //initialize weights from final hidden layer to output
        weight[numHLayers] = new Matrix(numOutputs, numHidden);

        //initialize bias matrices for hidden layers
        for (int i = 0; i < numHLayers; i++) {
            bias[i] = new Matrix(numH, 1);
        }

        //bias matrix for output layer
        bias[numHL] = new Matrix(numOutputs, 1);

        //randomize weights and biases
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
        output.setDataFromList(AFunctions.sigmoid(output.getDataAsList()));

        return output;
    }

    //returns a Matrix for the output of every layer
    public Matrix[] getLayerOutputs(float[] in) {
        Matrix input = new Matrix(numI, 1, in);
        Matrix[] outputs = new Matrix[numHL + 1];

        // gets first hidden layer
        outputs[0] = feedLayer(input, weight[0], bias[0]);

        // every remaining layer
        for (int i = 1; i < numHL + 1; i++) {
            outputs[i] = feedLayer(outputs[i - 1], weight[i], bias[i]);
        }

        return outputs;
    }

    //returns the output of the final layer
    public float[] getOutput(float[] input) {
        return getLayerOutputs(input)[numHL].getDataAsList();
    }

    //calculate the change in bias
    private Matrix getBiasGradient(Matrix weight, Matrix bias, Matrix nextNodeError, Matrix nextNodeOutput, Matrix previousNodeOutput) {
        Matrix deltaWeight = new Matrix(nextNodeOutput.getRows(), 1);
        deltaWeight.setDataFromList(AFunctions.dSigmoid(nextNodeOutput.getDataAsList()));
        deltaWeight = deltaWeight.schurProd(nextNodeError);
        deltaWeight = deltaWeight.schurProd(trainingCoef);

        return deltaWeight;
    }

    //calcuate the change in weights
    private Matrix getWeightGradient(Matrix previousNodeOutput, Matrix deltaBias) {
        return deltaBias.multiply(previousNodeOutput.transpose());
    }

    public void train(float[] inputs, float[] answer) {
        if (answer.length != numO) {
            throw new IllegalArgumentException("answer array has invalid length. Expected length: " + numO + ". Received: " + answer.length);
        }

        Matrix[] error = new Matrix[numHL + 1];
        Matrix[] outputs = getLayerOutputs(inputs);

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
            Matrix previousNodeOutput = i > 0 ? outputs[i - 1] : new Matrix(numI, 1, inputs);
            Matrix deltaBias = getBiasGradient(weight[i], bias[i], error[i], outputs[i], previousNodeOutput);
            Matrix deltaWeight = getWeightGradient(previousNodeOutput, deltaBias);
            bias[i] = bias[i].add(deltaBias);
            weight[i] = weight[i].add(deltaWeight);
        }
    }

    public void logAnswer(float[] input, float[] answer) {
        float[] rawOut = this.getOutput(input);
        int[] out = new int[rawOut.length];
        for (int i = 0; i < rawOut.length; i++) {
            out[i] = Math.round(rawOut[i]);
        }
        System.out.print("expected: " + Arrays.toString(answer));
        System.out.print(" actual: " + Arrays.toString(out));

        //TODO this wont work if there is more than one output
        System.out.println(" confidence: " + roundTo((Math.round(rawOut[0]) == 1 ? rawOut[0] : 1 - rawOut[0]), 3) * 100 + "%");
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

    static public final float roundTo(float val, int n) {
        float coef = (float)Math.pow(10, n);
        return (float)(Math.round(val * coef) / coef);
    }
}