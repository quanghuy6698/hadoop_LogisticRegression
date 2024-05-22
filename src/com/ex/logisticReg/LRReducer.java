package com.ex.logisticReg;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LRReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    private List<Double> predictions = new ArrayList<>();
    private List<Double> actuals = new ArrayList<>();

    @Override
    protected void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        for (DoubleWritable val : values) {
            if (key.toString().equals("prediction")) {
                predictions.add(val.get());
            } else if (key.toString().equals("actual")) {
                actuals.add(val.get());
            }
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if (predictions.size() != actuals.size()) {
            throw new IOException("Mismatch between number of predictions and actual values");
        }

        double ssTot = 0.0;
        double ssRes = 0.0;
        double meanY = 0.0;
        double mae = 0.0;
        double mse = 0.0;

        int n = actuals.size();
        for (double y : actuals) {
            meanY += y;
        }
        meanY /= n;

        for (int i = 0; i < n; i++) {
            double actual = actuals.get(i);
            double predicted = predictions.get(i);

            double error = actual - predicted;

            ssTot += Math.pow(actual - meanY, 2);
            ssRes += Math.pow(error, 2);
            mae += Math.abs(error);
            mse += error * error;
        }

        mae /= n;
        mse /= n;
        double rmse = Math.sqrt(mse);
        double r2 = 1 - (ssRes / ssTot);

        context.write(new Text("R2"), new DoubleWritable(r2));
        context.write(new Text("MAE"), new DoubleWritable(mae));
        context.write(new Text("MSE"), new DoubleWritable(mse));
        context.write(new Text("RMSE"), new DoubleWritable(rmse));
    }
}
