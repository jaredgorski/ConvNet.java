package convnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The CNN class encapsulates a convolutional neural network. This class contains all the primary logic for the
 * training, testing, and prediction mechanisms of the network. It particularly encapsulates the gradient descent
 * logic for minimizing cost when training the model, the recipe for processing the input data through various layers,
 * and the prediction engine for returning a given class with a specified confidence level.
 * @author Jared Gorski
 */
public class CNN {

    private String currentAction;
    private double learningRate;
    private Data inputData;
    private String[] classes;
    private double[] output;
    private List<Layer> layers = new ArrayList<>();
    private Schedule schedule = new Schedule();

    public List<Layer> getLayers() {
        return this.layers;
    }

    public void train(TrainingSet trainingSet) {
        setCurrentAction("train");
        setClasses(trainingSet.getClassMap());

        for (int i = 0; i < schedule.getIterations(); i++) {
            for (int j = 0; j < trainingSet.getDatasetLength(); j++) {
                trainEpoch(trainingSet.getDataByIndex(j));
            }
        }
    }

    private void trainEpoch(TrainingSet.TData tData) {
        Data tDataImg = tData.read();
        setInputData(tDataImg);
        feedforward();
        backpropagate(calcLoss(tData.getLabel()));
    }

    public double[] predict(double[][][] data) {
        setCurrentAction("predict");
        setInputData(data);
        feedforward();

        return this.output;
    }

    private void feedforward() {
        for (int i = 0; i < this.schedule.getLayers().length; i++) {
            Layer prevLayer;
            Layer currLayer;

            switch (this.schedule.getLayerByIndex(i)) {
                case "input":
                    if (i == 0 && this.layers.size() == 0) {
                        currLayer = new Layer().initInputLayer(inputData);
                        currLayer.setLayerIndex(i);
                        this.layers.add(currLayer);
                    }

                    break;
                case "conv":
                    prevLayer = this.layers.get(i - 1);

                    if (i >= this.layers.size()) {
                        currLayer = new Layer().initConvolutionalLayer(prevLayer);
                        currLayer.setLayerIndex(i);
                        currLayer.setNeuronsLength(32);
                        execConvolutionalLayer(currLayer);
                        this.layers.add(currLayer);
                    } else {
                        currLayer = this.layers.get(i);
                        currLayer.setLayerInput(prevLayer.getLayerOutput());
                        execConvolutionalLayer(currLayer);
                    }

                    break;
                case "pool":
                    prevLayer = this.layers.get(i - 1);

                    if (i >= this.layers.size()) {
                        currLayer = new Layer().initMaxPoolingLayer(prevLayer);
                        currLayer.setLayerIndex(i);
                        execMaxPoolingLayer(currLayer);
                        this.layers.add(currLayer);
                    } else {
                        currLayer = this.layers.get(i);
                        currLayer.setLayerInput(prevLayer.getLayerOutput());
                        execMaxPoolingLayer(currLayer);
                    }
                    break;
                case "activation":
                    prevLayer = this.layers.get(i - 1);

                    if (i >= this.layers.size()) {
                        currLayer = new Layer().initActivationLayer(prevLayer);
                        currLayer.setLayerIndex(i);
                        currLayer.setClasses(this.classes);
                        setOutput(activate(currLayer));
                        this.layers.add(currLayer);
                    } else {
                        currLayer = this.layers.get(i);
                        currLayer.setLayerInput(prevLayer.getLayerOutput());
                        setOutput(activate(currLayer));
                    }
                    break;
            }
        }
    }

    private void backpropagate(double loss) {
        int layersLength = this.schedule.getLayers().length;

        for (int i = 0; i < layersLength; i++) {
            int targetIndex = layersLength - i;
            Layer currLayer;
            Layer nextLayer;

            switch (this.schedule.getLayerByIndex(targetIndex)) {
                case "input":
                case "activation":
                    break;
                case "pool":
                    currLayer = this.layers.get(targetIndex);
                    nextLayer = this.layers.get(targetIndex + 1);
                    // backpropagatePool(currLayer, nextLayer);
                    break;
                case "conv":
                    currLayer = this.layers.get(targetIndex);
                    nextLayer = this.layers.get(targetIndex + 1);
                    backpropagateConv(currLayer, nextLayer);
                    break;
            }
        }
    }

    /**
     * Logic for a convolutional layer.
     *
     * @param layer
     */
    private void execConvolutionalLayer(Layer layer) {
        Data layerOutput = new Data();

        for (int i = 0; i < layer.getKernelsLength(); i++) {
            Data kernel = layer.getKernelByIndex(i);
            double bias = layer.getBiasByIndex(i);

            Data input = layer.getLayerInput();
            Data activationMap = this.evalValidConvolution(input, kernel, bias);
            layerOutput.stackMap(activationMap.read2d());
        }

        layer.setLayerOutput(layerOutput);
    }

    /**
     * Logic for a full convolution operation.
     *
     * @param input
     * @param kernel
     * @param bias
     * @return
     */
    private Data evalFullConvolution(Data input, Data kernel, double bias) {
        int m = input.getDimensions()[0];
        int n = input.getDimensions()[1];
        int o = input.getDimensions()[2];
        int km = kernel.getDimensions()[0];
        int kn = kernel.getDimensions()[1];
        int ko = kernel.getDimensions()[2];

        if (o != ko) {
            throw new RuntimeException("Malformed kernel and input. Different depths.");
        }

        double[][][] input3d = input.read3d();
        double[][][] fullInput = new double[m + 2 * (km - 1)][n + 2 * (kn - 1)][o];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < o; k++) {
                    fullInput[i + km - 1][j + kn - 1][k] = input3d[i][j][k];
                }
            }
        }

        Data fullInputData = new Data();
        fullInputData.write(fullInput);

        return evalValidConvolution(fullInputData, kernel, bias);
    }

    /**
     * Logic for a valid convolution operation.
     *
     * @param input
     * @param kernel
     * @param bias
     * @return
     */
    private Data evalValidConvolution(Data input, Data kernel, double bias) {
        int stride = 1;
        int m = input.getDimensions()[0];
        int n = input.getDimensions()[1];
        int o = input.getDimensions()[2];
        int km = kernel.getDimensions()[0];
        int kn = kernel.getDimensions()[1];
        int ko = kernel.getDimensions()[2];
        int kms = ((m - km) / stride) + 1;
        int kns = ((n - kn) / stride) + 1;

        if (o != ko) {
            throw new RuntimeException("Malformed kernel and input. Different depths.");
        }

        double[][] result = new double[kms][kns];

        for (int i = 0; i < kms; i++) {
            for (int j = 0; j < kns; j++) {
                int[] sliceH = new int[]{i, i + km};
                int[] sliceW = new int[]{j, j + kn};
                int[] sliceD = new int[]{0, ko};
                double[][][] inputSlice = input.slice3d(sliceH, sliceW, sliceD);
                result[i][j] = NumPute.dot(inputSlice, kernel.read3d()) + bias;
            }
        }

        Data activationMap = new Data();
        activationMap.write(result);
        return activationMap;
    }


    /**
     * Logic for a max pooling layer.
     * @param layer
     */
    private void execMaxPoolingLayer(Layer layer) {
        int size = layer.getPoolSize();
        int stride = layer.getPoolStride();

        Data input = layer.getLayerInput();
        Data poolOutput = this.evalMaxPooling(input, size, stride);
        layer.setLayerOutput(poolOutput);
    }

    /**
     * Logic for a max pooling operation on a tensor.
     * @param input
     * @return
     */
    private Data evalMaxPooling(Data input, int size, int stride) {
        int m = input.getDimensions()[0];
        int n = input.getDimensions()[1];
        int o = input.getDimensions()[2];
        int rm = ((m - size) / stride) + 1;
        int rn = ((n - size) / stride) + 1;

        double[][][] result = new double[rm][rn][o];

        for (int i = 0; i < rm; i = i + stride) {
            for (int j = 0; j < rn; j = j + stride) {
                for (int k = 0; k < o; k++) {
                    int si = i / stride;
                    int sj = j / stride;
                    List<Double> pool = new ArrayList<>();

                    for (int l = 0; l < size; l++) {
                        for (int p = 0; p < size; p++) {
                            pool.add(input.read3d()[i + l][j + p][k]);
                        }
                    }

                    result[si][sj][k] =  Collections.max(pool);
                }
            }
        }

        Data output = new Data();
        output.write(result);
        return output;
    }

    /**
     * Logic for the fully connected activation layer.
     * @param layer
     * @return
     */
    private double[] activate(Layer layer) {
        double[] output = new double[this.classes.length];

        execConvolutionalLayer(layer);
        Data convOutput = layer.getLayerOutput();
        int convOutputLength = convOutput.getDimensions()[2];

        if (this.classes.length != convOutputLength) {
            String message = String.format("Activation Error: Expected %d classes, evaluated %d output classes",
                    this.classes.length,
                    convOutputLength
            );

            throw new RuntimeException(message);
        }

        for (int i = 0; i < convOutput.getDimensions()[2]; i++) {
            output[i] = convOutput.read3d()[0][0][i];
        }

        output = NumPute.softmax(output);

        return output;
    }

    private double calcLoss(int[] trueLabel) {
        double loss = NumPute.crossEntropy(this.output, trueLabel);
        return loss;
    }

    /**
     * Logic for backpropagation on a convolutional layer.
     * @param layer
     * @param nextLayer
     */
    private void backpropagateConv(Layer layer, Layer nextLayer) {
        Data currLayerOutput = layer.getLayerOutput();
        int cm = currLayerOutput.getDimensions()[0];
        int cn = currLayerOutput.getDimensions()[1];
        int co = currLayerOutput.getDimensions()[2];

        int nErrLength = nextLayer.getLayerErrorsLength();

        if (co != nErrLength) {
            throw new RuntimeException("Backprop error: mismatched layer outputs vs nextLayer inputs");
        }

        for (int i = 0; i < co; i++) {
            int[] ch = new int[]{0, cm};
            int[] cw = new int[]{0, cn};
            int[] cd = new int[]{i, i + 1};
            double[][] cOutputMap = currLayerOutput.slice3d(ch, cw, cd)[0];

            double[][] nErrorMap = nextLayer.getLayerErrorByIndex(i).read2d();

            for (int k = 0; k < layer.getKernelsLength(); k++) {
                Data kernel = layer.getKernelByIndex(k);
                double bias = layer.getBiasByIndex(k);

                Data input = layer.getLayerInput();
                Data activationMap = this.evalValidConvolution(input, kernel, bias);
            }
        }
    }

    private void updateKernel() {
    }

    private void updateBias() {
    }

    /**
     * Set the schedule for the CNN to follow. Schedule should be an array of the desired layers, beginning with
     * an input layer, followed immediately by a convolutional layer, any hidden layers, and finishing with an
     * activation layer to determine the prediction matrix.
     * @param layers
     * @param iterations
     */
    private void setSchedule(String[] layers, int iterations) {
        this.schedule.setLayers(layers);
        this.schedule.setIterations(iterations);
    }

    /**
     * @param layers
     */
    private void setSchedule(String[] layers) {
        this.schedule.setLayers(layers);
    }

    private void setCurrentAction(String action) {
        switch (action) {
            case "train":
                this.currentAction = "train";
                break;
            case "predict":
                this.currentAction = "predict";
                break;
        }
    }

    public String getCurrentAction() {
        return this.currentAction;
    }

    public void setLearningRate(double rate) {
        this.learningRate = rate;
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    private void setClasses(String[] classes) {
        this.classes = classes;
    }

    private void setClasses(Map<Integer, String> classes) {
        String[] values = new String[classes.size()];

        int i = 0;
        for (Map.Entry<Integer, String> mapEntry : classes.entrySet()) {
            values[i] = mapEntry.getValue();
            i++;
        }

        this.classes = values;
    }

    private void setOutput(double[] output) {
        this.output = output;
    }

    private void setInputData(Data input) {
        this.inputData = input;
    }

    private void setInputData(double[][][] input) {
        Data newInput = new Data();
        newInput.write(input);
        this.inputData = newInput;
    }
}

