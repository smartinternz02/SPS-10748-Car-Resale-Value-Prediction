package comm.veda.ml;
import java.io.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.RandomForest;  
import weka.core.Instances;
public class MlModel {
	private static String getCurrPath() {
		String path=System.getProperty("user.dir")+"\\src\\main\\java\\comm\\veda\\ml";
		//System.out.println(path);
		return path.replace("\\","\\\\");
	}
	public static String trainPath="";
	public static String testPath="";
	private static void p6(Instances testDf, Instances trainDf) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("---------------------------------RANDOM FOREST-------------------------------");
		RandomForest rf=new RandomForest();
		rf.setNumIterations(200);
		Evaluation evaluation = new Evaluation(trainDf);
		evaluation.crossValidateModel(rf, testDf, 10, new Random(1));
		rf.buildClassifier(testDf);
		System.out.println(evaluation.toSummaryString("\nEvaluation Results\n******************************************************************************\n", true));
        System.out.println();
		System.out.println("S.No.\tTrueValue\tPredictedValue");
		for(int i=0;i<testDf.numInstances();i++) {
			String trueValue;
			trueValue=testDf.instance(i).toString(testDf.classIndex());
			double predValue=rf.classifyInstance(testDf.instance(i));
			System.out.println((i+1)+"\t"+trueValue+"\t"+predValue);
		}   
	}
	public static void main(String args[]) throws Exception{
		String trainFileName="\\\\train.arff";
		String testFileName="\\\\test1.arff";
		trainPath=getCurrPath()+trainFileName;
		//System.out.println(trainPath);
		testPath=getCurrPath()+testFileName;
		BufferedReader brTrain=new BufferedReader(new FileReader(trainPath));
		Instances trainDf=new Instances(brTrain);
		BufferedReader brTest=new BufferedReader(new FileReader(testPath));
		Instances testDf=new Instances(brTest);
		testDf.setClassIndex(testDf.numAttributes()-1);
		trainDf.setClassIndex(trainDf.numAttributes()-1);
		brTrain.close();
		brTest.close();
		p6(testDf,trainDf);
	}
	
	
}
