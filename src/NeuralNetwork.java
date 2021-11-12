public class NeuralNetwork {

    /**input to hidden*/
    private Matrix inputWeights;
    /**hidden to output*/
    private Matrix hiddenWeights;

    /**hidden layer bias*/
    private Matrix hiddenBias;
    /**output layer bias*/
    private Matrix outputBias;


    /** <p>Determines how quickly the network trains. Must be between 0 and 1 <p>
     * <p>Higher values = faster training but less precision<p>
     * <p>Lower values = slower training but better precision<p>
     */
    public float trainingCoef = 0.3f;

    private int numInputs;
    private int numHidden;
    private int numOutputs;

    public NeuralNetwork(int numInputs, int numHidden, int numOutputs) {

        inputWeights = new Matrix(numHidden, numInputs);
        hiddenWeights = new Matrix(numOutputs, numHidden);

        hiddenBias = new Matrix(numHidden, 1);
        outputBias = new Matrix(numOutputs, 1);

        hiddenBias.setAll(1f);
        outputBias.setAll(1f);

        inputWeights.randomize();
        hiddenWeights.randomize();
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutputs = numOutputs;
    }

    public Matrix[] getWeights() {
        return new Matrix[]{inputWeights, hiddenWeights};
    }

    public void logWeights() {
        System.out.println(inputWeights);
        System.out.println(hiddenWeights);
    }

    //used internally to get outputs of all levels in network so they can be used in the backpropegation algorithm
    private Matrix[] getAllOutputs(float[] input) {
        Matrix inputMatrix = new Matrix(numInputs, 1, input);

        //multiply weights matrix by input matrix
        Matrix hiddenMatrix = inputWeights.multiply(inputMatrix);
        //add bias
        hiddenMatrix = hiddenMatrix.add(hiddenBias);
        //apply sigmoid function
        hiddenMatrix.setDataFromList(sigmoid(hiddenMatrix.getDataAsList()));

        //same steps as above but for the output layer
        Matrix outputMatrix = hiddenWeights.multiply(hiddenMatrix);
        outputMatrix = outputMatrix.add(outputBias);
        outputMatrix.setDataFromList(sigmoid(outputMatrix.getDataAsList()));


        return new Matrix[]{hiddenMatrix, outputMatrix};
    }

    //Used by end user to receive neuralnet final outputs as an array
    public float[] getOutputs(float[] input) {
        return getAllOutputs(input)[1].getDataAsList();
    }

    public void train(float[] inputs, float[] answer) {
        if (answer.length != numOutputs) {
            throw new IllegalArgumentException("answer array has invalid length. Expected length: " + numOutputs + ". Received: " + answer.length);
        }

        //create matrices for input, hidden node output, and final output
        Matrix input = new Matrix(numInputs, 1, inputs);
        Matrix[] allOutputs = getAllOutputs(inputs);
        Matrix hiddenOutput = allOutputs[0];
        Matrix output = allOutputs[1];

        //calculate output error and store in matrix
        Matrix outputErr = new Matrix(numOutputs, 1);
        for(int i = 0; i < outputErr.getRows(); i++) {
            outputErr.data[i][0] = answer[i] - output.data[i][0];
        }
        Matrix hiddenErr = hiddenWeights.transpose().multiply(outputErr);

        //calculate change in weights for "hidden -> output" weights
        // (Lr)(Err)[Out(1 - s(Out)] * HOutT
        Matrix deltaHiddenWeights = new Matrix(numOutputs, 1, output.getDataAsList());
        deltaHiddenWeights.setDataFromList(dSigmoid(output.getDataAsList()));
        deltaHiddenWeights = deltaHiddenWeights.schurProd(outputErr);
        deltaHiddenWeights = deltaHiddenWeights.schurProd(trainingCoef);

        //adjust weight
        outputBias = outputBias.add(deltaHiddenWeights);
        hiddenWeights = hiddenWeights.add(deltaHiddenWeights.multiply(hiddenOutput.transpose()));

        //calculate change in weights for "input -> hidden" weights
        Matrix deltaInputWeights = new Matrix(numHidden, 1, hiddenOutput.getDataAsList());
        deltaInputWeights.setDataFromList(dSigmoid(hiddenOutput.getDataAsList()));
        deltaInputWeights = deltaInputWeights.schurProd(hiddenErr);
        deltaInputWeights = deltaInputWeights.schurProd(trainingCoef);

        //adjust weight
        hiddenBias = hiddenBias.add(deltaInputWeights);
        inputWeights = inputWeights.add(deltaInputWeights.multiply(input.transpose()));
    }

    private float[] sigmoid(float[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = sigmoid(inputs[i]);
        }
       return inputs;
    }

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

}
