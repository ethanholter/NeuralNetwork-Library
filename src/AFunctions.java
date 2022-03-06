public class AFunctions {

    public static final float activationFunction(String function, float x) {
        switch (function) {
            case "SIGMOID": return sigmoid(x);
            case "DSIGMOID": return dSigmoid(x);

            default: new IllegalArgumentException("Invalid activaton function name");
        }
        return 0f;
    }

    public static final float[] activationFunction(String function, float[] x) {
        switch (function) {
            case "SIGMOID": return sigmoid(x);
            case "DSIGMOID": return dSigmoid(x);

            default: new IllegalArgumentException("Invalid activaton function name");
        }
        return new float[]{0f};
    }
    
    private static final float sigmoid(float x) {
        return (float)(1f/(1f + Math.pow(Math.E, -x)));
    }
    
    private static final float dSigmoid(float x) {
        return x * (1 - sigmoid(x));
    }

    private static float[] sigmoid(float[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = sigmoid(inputs[i]);
        }
       return inputs;
    }

    private static float[] dSigmoid(float[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = dSigmoid(inputs[i]);
        }
       return inputs;
    }
}
