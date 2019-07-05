package convnet;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Data structure to store and manage image datasets for CNN training.
 * @author Jared Gorski
 */
public class TrainingSet {

    private List<TData> trainingData = new ArrayList<>();
    private String tsetRegex = "cnntset_\\[([\\d]+)/([\\d]+)\\]_";
    private Map<Integer, String> classMap = new HashMap<>();

    public void loadFiles(String dirname) throws IOException {
        File dir = new File(dirname);

        if (!dir.exists()) throw new FileNotFoundException("File not found");

        List<String> tsetFilepaths = getTSetPaths(dir);

        for (String path : tsetFilepaths) {
            TData intermediateTData = new TData();
            File file = new File(path);
            Data fileData = new Data();
            fileData.write(getRGBMatrixFromImgFile(file));
            intermediateTData.write(fileData);
            int[] oneHot = getOneHotLabelFromFile(file);
            intermediateTData.setLabel(oneHot);

            trainingData.add(intermediateTData);
        }
    }

    public List getDataset() {
        return this.trainingData;
    }

    public TData getDataByIndex(int index) {
        return this.trainingData.get(index);
    }

    public int getDatasetLength() {
        return this.trainingData.size();
    }

    public void setClassMap(int[] indices, String[] classes) {
        if (indices.length != classes.length) {
            throw new RuntimeException("Problem setting classmap: mismatched indices and classes.");
        }

        Map<Integer, String> intermediateClassMap = new HashMap<>();

        for (int i = 0; i < indices.length; i++) {
            intermediateClassMap.put(indices[i], classes[i]);
        }

        this.classMap = intermediateClassMap;
    }

    public Map getClassMap() {
        return this.classMap;
    }

    private List<String> getTSetPaths(final File dir) {
        List<String> paths = new ArrayList<>();

        for (final File entry : dir.listFiles()) {
            if (entry.exists()) {
                if (entry.isDirectory()) {
                    paths.addAll(getTSetPaths(entry));
                } else if (entry.getName().matches(this.tsetRegex)) {
                    paths.add(entry.getAbsolutePath());
                }
            }
        }

        return paths;
    }

    private int[] getOneHotLabelFromFile(File file) {
        int[] oneHot = new int[this.classMap.size()];
        String filename = file.getName();
        Pattern oneHotRegex = Pattern.compile(this.tsetRegex);
        Matcher filenameMatcher = oneHotRegex.matcher(filename);

        if (filenameMatcher.find()) {
            Pattern oneHotIndexPattern = Pattern.compile("([\\d+])");
            Matcher indexMatcher = oneHotIndexPattern.matcher(filenameMatcher.group(1));

            if (indexMatcher.find()) {
                int index = Integer.parseInt(indexMatcher.group(1));
                oneHot[index] = 1;
            }
        }

        return oneHot;
    }

    private double[][][] getRGBMatrixFromImgFile(File file) throws IOException {
        BufferedImage in = ImageIO.read(file);
        BufferedImage img = new BufferedImage(in.getWidth(), in.getHeight(), BufferedImage.TYPE_INT_ARGB);
        int imgHeight = img.getHeight();
        int imgWidth = img.getWidth();
        double[][][] rgbImg = new double[imgHeight][imgWidth][3];

        for (int y = 0; y < imgHeight; y++) {
            for (int x = 0; x < imgWidth; x++) {
                int color = img.getRGB(x, y);
                int r = (color & 0x00ff0000) >> 16;
                int g = (color & 0x0000ff00) >> 8;
                int b = color & 0x000000ff;

                rgbImg[x][y][0] = r;
                rgbImg[x][y][1] = g;
                rgbImg[x][y][2] = b;
            }
        }

        return rgbImg;
    }

    public class TData {
        private int[] label;
        private Data data;

        public void setLabel(int[] givenLabel) {
            this.label = givenLabel;
        }

        public int[] getLabel() {
            return this.label;
        }

        public void write(Data givenData) {
            this.data = givenData;
        }

        public Data read() {
            return this.data;
        }
    }
}
