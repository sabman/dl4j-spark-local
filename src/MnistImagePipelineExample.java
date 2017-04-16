import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * Created by shoaib on 4/16/17.
 */

public class MnistImagePipelineExample {
    // load data
    //  transform if needed
    public static Logger log = LoggerFactory.getLogger(MnistImagePipelineExample.class);

    public static void main(String[] args) throws Exception {
        // image information
        // 28*28 grayscale
        // grayscale = single channel
        int width = 28;
        int height = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 1;
        int outputNum = 10;

        File trainData = new File("/Users/shoaib/code/dl4j-spark-local/mnist_png/training/");
        File testData = new File("/Users/shoaib/code/dl4j-spark-local/mnist_png/testing/");

        // File split
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // tell it where to get the labels
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // tell it to read images
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);


        // initialize the recordReader
        // add a listener
        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        // Dataset Iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        DataNormalization scalar = new ImagePreProcessingScaler(0,1);
        scalar.fit(dataIter);
        dataIter.setPreProcessor(scalar);

//        for (int i = 0; i < 3; i++) {
//            DataSet ds = dataIter.next();
//            System.out.println(ds);
//            System.out.println(dataIter.getLabels());
//        }
        // Add neural net to the image pipeline
        // 1 Configure MultiLayerNetwork
        // 2 Train Model
        // 3 Test Model


    }
}
