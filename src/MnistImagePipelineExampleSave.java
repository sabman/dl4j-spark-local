import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
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

public class MnistImagePipelineExampleSave {
    // load data
    //  transform if needed
    public static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleSave.class);

    public static void main(String[] args) throws Exception {
        // image information
        // 28*28 grayscale
        // grayscale = single channel
        int width = 28;
        int height = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 128;
        int outputNum = 10;
        int numEpocs = 15;

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
//        recordReader.setListeners(new LogRecordListener());

        // Dataset Iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        DataNormalization scalar = new ImagePreProcessingScaler(0,1);
        scalar.fit(dataIter);
        dataIter.setPreProcessor(scalar);

        // Add neural net to the image pipeline
        // 1 Configure MultiLayerNetwork
        // 2 Train Model
        // 3 Test Model

        log.info("**** Build Model ****");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(width*height)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(100)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        log.info("**** Train Model ****");

        for (int i = 0; i < numEpocs; i++) {
            model.fit(dataIter);
        }

        log.info("**** Save Trained Model ****");

        File locationToSave = new File("trained_mnist_model.zip");

        // boolean save updater, if you wanna do further training you would save the updaters
        boolean saveUpdater = false;

        // ModelSerializer needs model name, weather to save updaters, filename
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        /*
        log.info("**** Evaluate Model ****");

        recordReader.reset();
        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(
                recordReader,
                batchSize,
                1,
                outputNum);
        scalar.fit(testIter);
        testIter.setPreProcessor(scalar);

        // create evaluation object
        Evaluation eval = new Evaluation(outputNum);
        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }

        log.info(eval.stats());
        */
    }
}
