public class AFunctions {
    
    public static final float sigmoid(float x) {
        return (float)(1f/(1f + Math.pow(Math.E, -x)));
    }
    
    public static final float dSigmoid(float x) {
        return x * (1 - sigmoid(x));
    }

    public static float[] sigmoid(float[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = sigmoid(inputs[i]);
        }
       return inputs;
    }

    public static float[] dSigmoid(float[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = dSigmoid(inputs[i]);
        }
       return inputs;
    }
}
