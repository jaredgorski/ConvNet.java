package convnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The Layer class represents a given layer of the CNN and provides methods for initiating and processing each layer
 * as the data is either trained or predicted upon.
 * @author Jared Gorski
 */
public class Layer {

    private String type; // Type of layer [input, conv, pool, activation]
    private int layerIndex; // Index number of layer, used to identify the layer in the layer stack.
    private boolean layerIndexSet = false; // Whether the layer has been indexed.
    private int neuronsLength; // Desired number of neurons for a convolutional layer.
    private String[] classes; // Expected output classes for activation layer.
    private Data layerInput = new Data(); // Input map stack before layer processing.
    private Data layerOutput = new Data(); // Output map stack after layer processing.
    private List<Data> kernels = new ArrayList<>(); // Weight maps for conv layer.
    private List<Double> biases = new ArrayList<>(); // Bias value(s) for conv layer.
    private List<Data> layerErrors = new ArrayList<>(); // Error maps for conv layer.
    // private int paddingWidth; // Width of any zero padding added to convolutions.
    private int poolSize = 2; // Size of the max pooling window; always square.
    private int poolStride = 2; // Stride of the max pooling window.
    private double cost; // Cost of current layer, sigmoid.

    /**
     * Initiate an input layer.
     * @param input
     * @return
     */
    public Layer initInputLayer(Data input) {
        Layer layer = new Layer();
        layer.setLayerInput(input);
        layer.setLayerType("input");

        return layer;
    }

    /**
     * Initiate a convolutional layer.
     * @param prevLayer
     * @return
     */
    public Layer initConvolutionalLayer(Layer prevLayer) {
        Layer layer = new Layer();
        layer.setLayerInput(prevLayer.getLayerOutput());
        layer.setLayerType("conv");

        int[] kernelShape = new int[]{5, 5, layer.getLayerInput().getDimensions()[2]};

        for (int i = 0; i < this.neuronsLength; i++) {
            Data newKernel = new Data();
            newKernel.write(NumPute.random3dMatrix(kernelShape));
            layer.setKernelByIndex(i, newKernel);
            double newBias = new Random().nextDouble();
            layer.setBiasByIndex(i, newBias);
        }

        return layer;
    }

    /**
     * Initiate a max pooling layer.
     * @param prevLayer
     * @return
     */
    public Layer initMaxPoolingLayer(Layer prevLayer) {
        Layer layer = new Layer();
        layer.setLayerInput(prevLayer.getLayerOutput());
        layer.setLayerType("pool");

        return layer;
    }

    /**
     * Initiate an activation layer.
     * @param prevLayer
     * @return
     */
    public Layer initActivationLayer(Layer prevLayer) {
        Layer layer = new Layer();
        layer.setLayerInput(prevLayer.getLayerOutput());
        layer.setLayerType("activation");

        int[] activationKernelShape = layer.getLayerInput().getDimensions();

        for (int i = 0; i < this.classes.length; i++) {
            Data activationKernel = new Data();
            activationKernel.write(NumPute.random3dMatrix(activationKernelShape));
            double activationBias = new Random().nextDouble();
            layer.setBiasByIndex(i, activationBias);
            layer.setKernelByIndex(i, activationKernel);
        }

        return layer;
    }

    /**
     * Set current layer type.
     * @param type
     */
    private void setLayerType(String type) {
        switch(type) {
            case "input":
                this.type = "input";
                break;
            case "conv":
                this.type = "conv";
                break;
            case "pool":
                this.type = "pool";
                break;
            case "activation":
                this.type = "activation";
                break;
            default:
                this.type = "input";
                break;
        }
    }

    /**
     */
    public String getLayerType() {
        return this.type;
    }

    /**
     */
    public void setLayerIndex(int index) {
        this.layerIndex = index;
        this.layerIndexSet = true;
    }

    /**
     */
    public boolean isLayerIndexSet() {
        return this.layerIndexSet;
    }

    /**
     */
    public int getLayerIndex() {
        return this.layerIndex;
    }

    /**
     * External method for storing input of a given layer.
     * @param layerInput
     */
    public void setLayerInput(Data layerInput) {
        this.layerInput = layerInput;
    }

    /**
     * @return
     */
    public Data getLayerInput() {
        return this.layerInput;
    }

    /**
     * External method for storing result of a given layer.
     * @param layerOutput
     */
    public void setLayerOutput(Data layerOutput) {
        this.layerOutput = layerOutput;
    }

    /**
     */
    public Data getLayerOutput() {
        return this.layerOutput;
    }

    /**
     */
    public void setNeuronsLength(int length) {
        this.neuronsLength = length;
    }

    /**
     */
    public void setClasses(String[] classes) {
        this.classes = classes;
    }

    /**
     * Define a new kernel for a convolutional layer.
     * @param index
     * @param kernel
     */
    public void setKernelByIndex(int index, Data kernel) {
        this.kernels.set(index, kernel);
    }

    /**
     */
    public Data getKernelByIndex(int index) {
        return this.kernels.get(index);
    }

    /**
     */
    public int getKernelsLength() {
        return this.kernels.size();
    }

    /**
     */
    public void setBiasByIndex(int index, double bias) {
        this.biases.set(index, bias);
    }

    /**
     */
    public double getBiasByIndex(int index) {
        return this.biases.get(index);
    }

    /**
     * Define a new error map for a convolutional layer.
     * @param index
     * @param errorData
     */
    public void setLayerErrorByIndex(int index, Data errorData) {
        this.layerErrors.set(index, errorData);
    }

    /**
     */
    public Data getLayerErrorByIndex(int index) {
        return this.layerErrors.get(index);
    }

    /**
     */
    public int getLayerErrorsLength() {
        return this.layerErrors.size();
    }

    /**
     * @param newSize
     */
    public void setPoolSize(int newSize) {
        this.poolSize = newSize;
    }

    /**
     * @return
     */
    public int getPoolSize() {
        return this.poolSize;
    }

    /**
     * @param newStride
     */
    public void setPoolStride(int newStride) {
        this.poolStride = newStride;
    }

    /**
     * @return
     */
    public int getPoolStride() {
        return this.poolStride;
    }
}

