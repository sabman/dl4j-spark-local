import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Created by shoaib on 4/16/17.
 */
public class MnistImagePipelineLoadChooser {
    public static Logger logger = LoggerFactory.getLogger(MnistImagePipelineLoadChooser.class);

    public static String fileChose(){
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);

        if (ret == JFileChooser.APPROVE_OPTION){
            File file = fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;
        }
        else {
            return null;
        }
    }

    public static void main(String[] args) throws Exception {
        int height = 28;
        int width = 28;
        int channels = 1;

        List<Integer> labelList = Arrays.asList(2,3,7,1,6,4,0,5,8,9);

        String filechose = fileChose().toString();

        // load model
        File locationToSave = new File("trained_mnist_model.zip");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        logger.info("**** test your image against saved network ****");

        // FileChoose is a string so we need a file (get file)
        File file = new File(filechose);

        // load image into matrix of numerical data

        NativeImageLoader loader = new NativeImageLoader(height, width,channels);
        INDArray image = loader.asMatrix(file);

        // scale values 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        // pass through neural net
        INDArray output = model.output(image);


        logger.info("## File Chosen: " + filechose);
        logger.info("## The Neural Net Prediction ##");
        logger.info("## list of probabilities per label ##");
        logger.info("## list of labels in Order ##");
        logger.info(output.toString());
        logger.info(labelList.toString());
    }
}
