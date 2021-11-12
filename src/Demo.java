import java.util.Arrays;

public class Demo {
    public static void main(String[] args) {
        NeuralNetwork brain = new NeuralNetwork(2, 4, 1);
        brain.trainingCoef = 0.1f;
        testXOR(brain);

    }

    public static void testXOR(NeuralNetwork brain) {
        float[] inputs = new float[2];
        float[] answers = new float[1];

        for (int i = 0; i < 100000; i++) {
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = (int)(Math.random() * 2);
            }
            answers[0] = (inputs[0] == 1 || inputs[1] == 1) && !(inputs[0] == 1 && inputs[1] == 1) ? 1 : 0;
            brain.train(inputs, answers);
        }

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = (int)(Math.random() * 2);
            }
            float out = brain.getOutputs(inputs)[0];

            System.out.println(Arrays.toString(inputs));
            System.out.println(out);
            
        }
    }
}
