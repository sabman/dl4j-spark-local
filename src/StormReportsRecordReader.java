import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

import java.util.Date;
import java.util.List;

/**
 * Created by shoaib on 4/5/17.
 */
public class StormReportsRecordReader {
    public static void main(String[] args) {
        int numLinesToSkip = 0;
        String delimiter = ",";
        String baseDir = "/Users/shoaib/code/dl4j-spark-local/";
        String fileName = "input/reports.csv";
        String inputPath = baseDir + fileName;
        String timeStamp = String.valueOf(new Date().getTime());
        String outputPath = baseDir + "output/reports_processed_" + timeStamp;

//  CSV sample:
//  161006-1655,UNK,2 SE BARTLETT,LABETTE,KS,37.03,-95.19,TRAINED SPOTTER REPORTS TORNADO ON THE GROUND. (ICT),TOR
//  161006-1712,UNK,2 ESE BARTLETT,LABETTE,KS,37.05,-95.18,NUMEROUS TREE LIMBS DOWN FROM A BRIEF TORNADO TOUCHDOWN. CORRECTED FOR TIME. (ICT),TOR
//  161006-1720,UNK,2 ESE BARTLETT,LABETTE,KS,37.05,-95.18,NUMEROUS TREE LIMBS DOWN FROM A BRIEF TORNADO TOUCHDOWN. (ICT),TOR
//      Fields are: datetime,severity,location,county,state,lat,lon,comment,type

        /*
         * Define input schema
         */
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("datetime","severity","location","county","state")
                .addColumnsDouble("lat","lon")
                .addColumnsString("comment")
                .addColumnCategorical("type", "TOR", "WIND", "HAIL")
                .build();

        /**
         * Transform step that converts categorical data to integers
         * and extracts lat/lon ...
         */
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("datetime","severity","location","county","state", "comment")
                .categoricalToInteger("type")
                .build();

        int numActions = tp.getActionList().size();
        for (int i = 0; i < numActions; i++) {
            System.out.println("\n\n==================================");
            System.out.println("-- Schema after step " + i +
                " (" + tp.getActionList().get(i) + ") --"
            );
            System.out.println(tp.getSchemaAfterStep(i));
        }

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Storm Reports Record Reader Transfrom");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        /**
         * get our data into spark RDDs
         */
        // read the data file
        JavaRDD<String> lines = sc.textFile(inputPath);
        // convert to Writable
        JavaRDD<List<Writable>> stormReports = lines.map(new StringToWritablesFunction(new CSVRecordReader()));
        // run our transform process
        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(stormReports,tp);
        // convert Writable back to string for export
        JavaRDD<String> toSave= processed.map(new WritablesToStringFunction(","));

        toSave.saveAsTextFile(outputPath);
    }
}
