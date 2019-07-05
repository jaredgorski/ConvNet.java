package convnet;

/**
 * The Data class includes a set of methods for interacting with the data currently being operated on.
 * This includes methods to write new data, read existing data, clear existing data, save existing data
 * to a file, and return supervised training information for the CNN.
 * @author Jared Gorski
 */
public class Data {

    private int[] dimensions = new int[3]; // [ Height, Width, Depth ]
    private double[][] data2d = null;
    private double[][][] data3d = null;

    /**
     * Stack current data on top of a 2d map of same dimensions.
     * @param input
     */
    public void stackMap(double[][] input) {
        int m = input.length;
        int n = input[0].length;

        if (this.data2d == null && this.data3d == null) {
            this.write(input);
        } else {

            if (m != this.dimensions[0] || n != this.dimensions[1]) {
                throw new RuntimeException("Cannot stack odd-sized map.");
            } else if (this.dimensions[2] == 0) {
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        this.data3d[i][j][0] = this.data2d[i][j];
                    }
                }

                this.data2d = null;
                this.setDimensions3d();
            }

            int stackIndex = this.dimensions[2]++;

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    this.data3d[i][j][stackIndex] = input[i][j];
                }
            }
        }
    }

    /**
     * Returns a 3d slice of the data for operation.
     * @param sliceHeightPair
     * @param sliceWidthPair
     * @param sliceDepthPair
     * @return slice
     */
    public double[][][] slice3d(int[] sliceHeightPair, int[] sliceWidthPair, int[] sliceDepthPair) {
        int sliceH = sliceHeightPair[1] - sliceHeightPair[0];
        int sliceW = sliceWidthPair[1] - sliceWidthPair[0];
        int sliceD = sliceDepthPair[1] - sliceDepthPair[0];
        double[][][] slice = new double[sliceH][sliceW][sliceD];

        if (data3d == null) {
            throw new RuntimeException("No data to slice.");
        } else if (sliceHeightPair[1] < sliceHeightPair[0] ||
                sliceWidthPair[1] < sliceWidthPair[0] ||
                sliceDepthPair[1] < sliceDepthPair[0]) {
            throw new RuntimeException("Malformed pairs.");
        } else if (sliceHeightPair[1] > dimensions[0] ||
                sliceWidthPair[1] > dimensions[1] ||
                sliceDepthPair[1] > dimensions[2]) {
            throw new RuntimeException("Slice exceeds dimensions.");
        } else {
            for (int i = 0; i < sliceH; i++) {
                for (int j = 0; j < sliceW; j++) {
                    for (int k = 0; k < sliceD; k++) {
                        int coordI = sliceHeightPair[0] + i;
                        int coordJ = sliceHeightPair[0] + j;
                        int coordK = sliceHeightPair[0] + k;

                        slice[coordI][coordJ][coordK] = data3d[coordI][coordJ][coordK];
                    }
                }
            }
        }

        return slice;
    }

    /**
     */
    private void setDimensions2d() {
        if (this.data2d != null) {
            int[] shape = NumPute.shape(this.data2d);
            this.dimensions = new int[]{shape[0], shape[1], 0};
        } else {
            throw new RuntimeException("No data. Cannot set dimensions.");
        }
    }

    /**
     */
    private void setDimensions3d() {
        if (this.data3d != null) {
            this.dimensions = NumPute.shape(this.data3d);
        } else {
            throw new RuntimeException("No data. Cannot set dimensions.");
        }
    }

    /**
     * Returns the dimensions of the current data.
     * @return
     */
    public int[] getDimensions() {
        return this.dimensions;
    }

    /**
     * Nullifies the current data variable.
     */
    public void nullify() {
        if (this.dimensions[2] == 0) {
            this.data2d = null;
        } else {
            this.data3d = null;
        }
    }

    /**
     * Returns the current 2d data.
     * @return
     */
    public double[][] read2d() {
        return this.data2d;
    }

    /**
     * Returns the current 3d data.
     * @return
     */
    public double[][][] read3d() {
        return this.data3d;
    }

    /**
     * Saves the current data to a file for future reference.
     */
    public void save() {}

    /**
     * Writes new 2d data for operation.
     * @param newData
     */
    public void write(double[][] newData) {
        if (this.data2d == null) {
            this.data2d = newData;
            setDimensions2d();
        } else {
            throw new RuntimeException("Data already exists. Cannot overwrite.");
        }
    }

    /**
     * Writes new 3d data for operation.
     * @param newData
     */
    public void write(double[][][] newData) {
        if (this.data3d == null) {
            this.data3d = newData;
            setDimensions3d();
        } else {
            throw new RuntimeException("Data already exists. Cannot overwrite.");
        }
    }
}

