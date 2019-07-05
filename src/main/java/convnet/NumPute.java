package convnet;

import java.util.Arrays;
import java.util.Random;

/**
 * The cnn.NumPute class contains numpy-like operations necessary to the convnet.CNN class (and otherwise).
 * @author Jared Gorski
 */
public class NumPute {

    /**
     * Dot product of two 2d matrices.
     * @param a
     * @param b
     * @return
     */
    public static double dot(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;

        if (n1 != n2 || m1 != m2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }

        double result = 0.0;

        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n1; j++) {
                result += a[i][j] * b[i][j];
            }
        }

        return result;
    }

    /**
     * Dot product of two 3d matrices.
     * @param a
     * @param b
     * @return
     */
    public static double dot(double[][][] a, double[][][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int o1 = a[0][0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        int o2 = b[0][0].length;

        if (n1 != n2 || m1 != m2 || o1 != o2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }

        double result = 0.0;

        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n1; j++) {
                for (int k = 0; k < o1; k++) {
                    result += a[i][j][k] * b[i][j][k];
                }
            }
        }

        return result;
    }

    public static double[][] rot180(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n / 2; j++) {
                b[i][j] = a[i][n - 1 - j];
                b[i][n - 1 - j] = a[i][j];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m / 2; j++) {
                b[i][j] = a[m - 1 - i][j];
                b[m - 1 - i][j] = a[i][j];
            }
        }

        return b;
    }

    public static double[][] scale2d(double[][] a, int[] scale) {
        int m = a.length;
        int n = a[0].length;
        int ms = scale[0];
        int ns = scale[1];

        double[][] result = new double[m * ms][n * ns];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < ms; k++) {
                    for (int l = 0; l < ns; l++) {
                        result[(i * ms) + k][(j * ns) + l] = a[i][j];
                    }
                }
            }
        }

        return result;
    }

    public static double[][] kronecker2d(double[][] a, double[][] b) {
        int ma = a.length;
        int na = a[0].length;
        int mb = b.length;
        int nb = b[0].length;

        double[][] result = new double[ma * mb][na * nb];

        for (int i = 0; i < ma; i++) {
            for (int j = 0; j < na; j++) {
                for (int k = 0; k < mb; k++) {
                    for (int l = 0; l < nb; l++) {
                        result[(i * ma) + k][(j * mb) + l] = a[i][j] * b[k][l];
                    }
                }
            }
        }

        return result;
    }

    public static void print2dMatrix(double[][] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.println(Arrays.toString(a[i]));
        }
    }

    /**
     * Returns a random 3d matrix of given shape.
     * @param shape
     * @return
     */
    public static double[][][] random3dMatrix(int[] shape) {
        int m = shape[0];
        int n = shape[1];
        int o = shape[2];
        double[][][] result = new double[m][n][o];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < o; k++) {
                    result[i][j][k] = new Random().nextDouble();
                }
            }
        }

        return result;
    }

    /**
     * Returns shape of matrix as an array.
     * @param a
     * @return
     */
    public static int[] shape(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        int[] shape = {m, n};

        return shape;
    }

    /**
     * Returns shape of matrix as an array.
     * @param a
     * @return
     */
    public static int[] shape(double[][][] a) {
        int m = a.length;
        int n = a[0].length;
        int o = a[0][0].length;
        int[] shape = {m, n, o};

        return shape;
    }

    /**
     * Cross entropy function for calculating loss.
     * @param x
     * @param y
     */
    public static double crossEntropy(double[] x, int[] y) {
        double acc = 0.0;

        for (int i = 0; i < y.length; i++) {
            if (x[i] != 0) {
                acc += (y[i] * Math.log(x[i]));
            }
        }

        double loss = -acc;
        return loss;
    }

    /**
     * SoftMax function for double.
     * @param x
     * @return
     */
    public static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < x.length; i++) {
            if (x[i] > max) {
                max = x[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double out = Math.exp(x[i] - max);
            x[i] = out;
            sum += out;
        }

        for (int i = 0; i < x.length; i++) {
            x[i] /= sum;
        }

        return x;
    }

    public static double[][][] deriveMap(double[][][] x) {
        return x;
    }

    public static double matrixTotal(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double sum = 0.0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum += a[i][j];
            }
        }

        return sum;
    }

    public static double matrixTotal(double[][][] a) {
        int m = a.length;
        int n = a[0].length;
        int o = a[0][0].length;
        double sum = 0.0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < o; k++) {
                    sum += a[i][j][k];
                }
            }
        }

        return sum;
    }

    public static double[][] elementOperate(double[][] a, double b, String operation) {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n];


        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = operate(a[i][j], b, operation);
            }
        }

        return result;
    }

    public static double[][][] elementOperate(double[][][] a, double b, String operation) {
        int m = a.length;
        int n = a[0].length;
        int o = a[0][0].length;
        double[][][] result = new double[m][n][o];


        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < o; k++) {
                    result[i][j][k] = operate(a[i][j][k], b, operation);
                }
            }
        }

        return result;
    }

    private static double operate(double a, double b, String operation) {
        double result = 0.0;

        switch(operation) {
            case "add":
                result = a + b;
                break;
            case "subtract":
                result = a - b;
                break;
            case "multiply":
                result = a * b;
                break;
            case "divide":
                result = a / b;
                break;
        }

        return result;
    }

    /**
     * Sigmoid function for double.
     * @param x
     * @return
     */
    public static double sigmoid(double x) {
        return (1 / (1 + Math.pow(Math.E, (-1 * x))));
    }

    /**
     * Sigmoid function for matrix.
     * @param a
     * @return
     */
    public static double[][][] sigmoid(double[][][] a) {
        int m = a.length;
        int n = a[0].length;
        int o = a[0][0].length;
        double[][][] result = new double[m][n][o];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < o; k++) {
                    result[i][j][k] = (1.0 / (1 + Math.exp(-a[i][j][k])));
                }
            }
        }

        return result;
    }

    /**
     * Sum of squares error on 1d result vector against 1d result vector from supervised training data.
     * @param y
     * @param yHat
     * @return
     */
    public static double sse(double[] y, double[] yHat) {
        int yL = y.length;
        int yHatL = yHat.length;
        double sumSq = 0;

        if (yL != yHatL) {
            throw new RuntimeException("Cannot determine error of unmatched matrices.");
        }

        for (int i = 0; i < yL; i++) {
            sumSq += Math.pow((y[i] - yHat[i]), 2);
        }

        return sumSq;
    }

    /**
     * Returns a transpose of the input matrix.
     * @param a
     * @return
     */
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = a[i][j];
            }
        }

        return result;
    }

    /**
     * Returns a matrix of given shape filled with zeros.
     * @param shape
     * @return
     */
    public static double[][] zeros(int[] shape) {
        int m = shape[0];
        int n = shape[1];
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = 0;
            }
        }

        return result;
    }
}

