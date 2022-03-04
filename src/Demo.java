public class Demo {

    static int iterations = 100000;
    public static void main(String[] args) {
        long startTime = System.nanoTime();

        NeuralNetwork Brain = new NeuralNetwork(2, 8, 1, 1);
        Brain.trainingCoef = 0.1f;
        
        testXOR(Brain);
        
        long elapsedNanos = System.nanoTime() - startTime;
        System.out.println("finished after " + elapsedNanos/1000000 + " miliseconds");
        System.out.println("\n" + Brain);
    }

    public static float xor(float a_, float b_) {
        boolean a = (a_ == 1);
        boolean b = (b_ == 1);
        return !(a && b) && (a || b) ? 1f : 0f;
    }

    public static void testXOR(NeuralNetwork brain) {
        float[] inputs = new float[2];
        float[] answers = new float[1];

        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = (int)(Math.random() * 2);
            }
            answers[0] = xor((int)inputs[0], (int)inputs[1]);
            brain.train(inputs, answers);
        }

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = (int)(Math.random() * 2);
            }
            brain.logAnswer(inputs, new float[] {xor(inputs[0], inputs[1])});

        }
    }

    public static void testSIN(NeuralNetwork brain) {
        float[] answers = new float[1];

        for (int i = 0; i < iterations; i++) {
            float angle = (float)(Math.random() * 2 * Math.PI);
            float[] input =  {map(angle, -1, 1, 0, 1)};
            answers[0] = (float)Math.sin(angle);
            brain.train(input, answers);
        }

        for (int i = 0; i < 5; i++) {

            float angle = (float)(Math.random() * 2 * Math.PI);
            float[] input =  {map(angle, -1, 1, 0, 1)};
            float out = brain.getOutputs(input)[0];
            System.out.println(Math.sin(angle));
            System.out.println(map(out, 0, 1, -1, 1) + "\n");
        }

        float sum = 0;
        for (int i = 0; i < 1000; i++) {
            float angle = (float)(Math.random() * 2 * Math.PI);
            float[] input =  {map(angle, -1, 1, 0, 1)};
            float out = brain.getOutputs(input)[0];
            float error = (float)Math.abs(Math.sin(angle) - out);
            sum += error;
        }
        float averageErr = sum/1000;
        System.out.println("average error: " + averageErr);
        
    }

    // maps a number from one range to another
    static public final float map(float value, float istart, float istop, float ostart, float ostop) {
        return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
    }
}
