package convnet;

/**
 * The Schedule class provides a structure for the execution plan of the CNN.
 * @author Jared Gorski
 */
public class Schedule {

    private String[] layers;
    private int iterations;

    public String[] getLayers() {
        return this.layers;
    }

    public int getIterations() {
        return this.iterations;
    }

    public String getLayerByIndex(int i) {
        return this.layers[i];
    }

    public void setLayers(String[] layers) {
        this.layers = layers;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }
}
