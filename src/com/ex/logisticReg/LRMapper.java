package com.ex.logisticReg;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class LRMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {
    private double[] weights;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        // Initialize weights here or fetch from configuration
        weights = new double[]{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}; // Example weights initialization
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] parts = line.split(",");
        
        double y = Double.parseDouble(parts[7]);
        double[] x = new double[parts.length - 1];

        for (int i = 0; i < parts.length - 1; i++) {
            x[i] = Double.parseDouble(parts[i]);
        }

        double prediction = predict(x);
        context.write(new Text("prediction"), new DoubleWritable(prediction));
        context.write(new Text("actual"), new DoubleWritable(y));
    }

    private double predict(double[] x) {
        double z = 0.0;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * x[i];
        }
        return 1 / (1 + Math.exp(-z)); // Sigmoid function
    }
}
