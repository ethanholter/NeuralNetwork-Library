import java.lang.Math;

/**
 * Container object for a mathematical matrix
 * 
 * data
 * getCols()
 * getRows()
 */
public class Matrix {
    private int rows = 0;
    private int cols = 0;
    public float[][] data;

    Matrix(int rows, int cols) {
        if(rows == 0 || cols == 0) {
            throw new IllegalArgumentException("Matrix cannot have a dimension of 0");
        } 

        this.rows = rows;
        this.cols = cols;

        //initialize array of 0s
        this.data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this.data[i][j] = 0;
            }
        }
    }

    Matrix(int rows, int cols, float[][] data) {
        if(rows == 0 || cols == 0) {
            throw new IllegalArgumentException("Matrix cannot have a dimension of 0");
        } 

        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];
        //creates a deep copy of the array because java is stupid and wont let me use .clone()
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this.data[i][j] = data[i][j];
            }
        }
    }

    Matrix(int rows, int cols, float[] list) {
        this(rows, cols, getDataFromList(rows, cols, list));
    }

    public Matrix randomize() {
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this.data[i][j] = (float)(Math.random() * 2 - 1);
            }
        }
        return this;
    }

    public Matrix randomize(int num) {
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this.data[i][j] = (int)(Math.random() * num);
            }
        }
        return this;
    }

    public Matrix setAll(float num) {
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this.data[i][j] = num;
            }
        }
        return this;
    }

    /**
     * Switches a matrix's rows and columns. (turned 90 degrees)
     * @return New Matrix object
     */
    public Matrix transpose() {
        Matrix newMatrix = new Matrix(this.cols, this.rows);
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newMatrix.data[j][i] = this.data[i][j];
            }
        }
        return newMatrix;
    }

    /**
     * Adds both matrices element-wise.
     * @param matrix to be added
     * @return new Matrix object
     */
    public Matrix add(Matrix matrix) {
        Matrix newMatrix = new Matrix(this.rows, this.cols, this.data);
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newMatrix.data[i][j] += matrix.data[i][j];
            }
        }
        return newMatrix;
    }
    
    /**
     * Adds a constant to each element in the matrix.
     * @param num to be added
     * @return new Matrix object
     */
    public Matrix add(float num) {
        Matrix newMatrix = new Matrix(this.rows, this.cols, this.data);
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newMatrix.data[i][j] += num;
            }
        }
        return newMatrix;
    }

    /**
     * Multiplies each element in the matrix by a constant.
     * @param num to be multiplied
     * @return new Matrix object
     */
    public Matrix schurProd(float num) {
        Matrix newMatrix = new Matrix(this.rows, this.cols, this.data);
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newMatrix.data[i][j] *= num;
            }
        }
        return newMatrix;
    }

    public Matrix schurProd(Matrix b) {
        Matrix newMatrix = new Matrix(this.rows, this.cols, this.data);
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newMatrix.data[i][j] *= b.data[i][j];
            }
        }
        return newMatrix;
    }

    /**
     * 
     * @param matrix second matrix to be multiplied
     * @return new Matrix object
     */
    public Matrix multiply(Matrix matrix) {
        Matrix newMatrix = new Matrix(this.getRows(), matrix.getCols());
        if(this.getCols() != matrix.getRows()) {
            throw new IllegalArgumentException("Matrices are not compatable\n" + this + "\n" + matrix);
        }
        float sum = 0;
        for (int i = 0; i < newMatrix.getRows(); i++) {
            for(int j = 0; j < newMatrix.getCols(); j++) {
                for(int k = 0; k < this.getCols(); k++) {
                    sum += this.data[i][k] * matrix.data[k][j];
                }
                newMatrix.data[i][j] = sum;
                sum = 0;
            }
        }
        return newMatrix;
    }
    
    public final Matrix round(int numDecimals) {
        Matrix newMatrix = new Matrix(this.getRows(), this.getCols());
        float coef = (float)Math.pow(10, numDecimals);
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                newMatrix.data[i][j] = (float)(Math.round(this.data[i][j] * coef) / coef);
            }
        }
        return newMatrix;
    }

    // this is kind of unnecessary but this adds a static version of all methods
    public static Matrix transpose(Matrix a) {
        return a.transpose();
    }
    
    public static Matrix add(Matrix a, Matrix b) {
        return a.add(b);
    }
    
    public static Matrix add(Matrix a, int num) {
        return a.add(num);
    }
    
    public static Matrix schurProd(Matrix a, int num) {
        return a.schurProd(num);
    }

    public static Matrix schurProd(Matrix a, Matrix b) {
        return a.schurProd(b);
    }
    
    public static Matrix multiply(Matrix a, Matrix b) {
        return a.multiply(b);
    }

    public static Matrix round(Matrix a, int num) {
        return a.round(num);
    }

    public static float[][] getDataFromList(int rows, int cols, float[] list) throws ArrayIndexOutOfBoundsException {
        try {
            float[][] data = new float[rows][cols];
            int count = 0;
            for (int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    data[i][j] = list[count];
                    count++;
                }
            }
            return data;
        } catch (Exception e) {
            System.out.println("Expected array of length " + (rows * cols) + " but recieved array of length " + list.length);
            throw e;
        }
    }
    
    public int getRows() {
        return this.rows;
    }
    
    public int getCols() {
        return this.cols;
    }

    public float[] getDataAsList() {
        float[] output = new float[this.rows * this.cols];
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                output[i * this.cols + j] = this.data[i][j];
            }
        }
        return output;
    }

    public void setDataFromList(float[] list) {
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this.data[i][j] = list[i * this.cols + j];
            }
        }
    }
    
    public String toString() {
        // String output = this.rows + "x" + this.cols + " Matrix:\n";
        String output = "";
        for (int i = 0; i < this.rows; i++) {
            output = output.concat("[");
            for(int j = 0; j < this.cols; j++) {
                output = output.concat(this.data[i][j] + (j != this.cols - 1 ? ", ": ""));
            }
            output = output.concat("]\n");
        }
        return output;
    }
}
