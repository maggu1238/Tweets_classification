package cc.mallet.classify;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.logging.Logger;

import cc.mallet.optimize.Optimizable;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.MalletProgressMessageLogger;
import cc.mallet.util.Maths;

/**
 * Training of MaxEnt models with labeled features using
 * Generalized Expectation Criteria.
 * 
 * Based on: 
 * "Learning from Labeled Features using Generalized Expectation Criteria"
 * Gregory Druck, Gideon Mann, Andrew McCallum
 * SIGIR 2008
 * 
 * @author Gregory Druck <a href="mailto:gdruck@cs.umass.edu">gdruck@cs.umass.edu</a>
 */

/**
 * @author gdruck
 *
 */
public class MaxEntOptimizableByGE implements Optimizable.ByGradientValue {
  
  private static Logger progressLogger = MalletProgressMessageLogger.getLogger(MaxEntOptimizableByLabelLikelihood.class.getName()+"-pl");

  private boolean cacheStale = true;
  private boolean useValues;
  private int defaultFeatureIndex;
  private double temperature;
  private double objWeight;
  private double cachedValue;
  private double gaussianPriorVariance;
  private double[] cachedGradient;
  private double[] parameters;
  private InstanceList trainingList;
  private MaxEnt classifier;
  private HashMap<Integer,double[]> constraints;
  private HashMap<Integer,Integer> mapping;
  
  /**
   * @param trainingList List with unlabeled training instances.
   * @param constraints Feature expectation constraints.
   * @param initClassifier Initial classifier.
   */
  public MaxEntOptimizableByGE(InstanceList trainingList, HashMap<Integer,double[]> constraints, MaxEnt initClassifier) {
    useValues = false;
    temperature = 1.0;
    objWeight = 1.0;
    this.trainingList = trainingList;
    
    int numFeatures = trainingList.getDataAlphabet().size();
    defaultFeatureIndex = numFeatures;
    int numLabels = trainingList.getTargetAlphabet().size();
    parameters = new double[(numFeatures + 1) * numLabels];
    cachedGradient = new double[(numFeatures + 1) * numLabels];
    cachedValue = 0;
       
    if (classifier != null) {
      this.classifier = initClassifier;
    }
    else {
      this.classifier = new MaxEnt(trainingList.getPipe(),parameters);
    }
    
     this.constraints = constraints;
  } 
  
  /**
   * Sets the variance for Gaussian prior or
   * equivalently the inverse of the weight 
   * of the L2 regularization term.
   * 
   * @param variance Gaussian prior variance.
   */
  public void setGaussianPriorVariance(double variance) {
    this.gaussianPriorVariance = variance;
  }
  
  
  /**
   * Set the temperature, 1 / the exponent model predicted probabilities 
   * are raised to when computing model expectations.  As the temperature
   * increases, model probabilities approach 1 for the maximum probability
   * class, and 0 for other classes.  DEFAULT: 1  
   * 
   * @param temp Temperature.
   */
  public void setTemperature(double temp) {
    this.temperature = temp;
  }
  
  /**
   * The weight of GE term in the objective function.
   * 
   * @param weight GE term weight.
   */
  public void setWeight(double weight) {
    this.objWeight = weight;
  }
  
  public MaxEnt getClassifier() {
    return classifier;
  }
  
  public double getValue() {   
    if (!cacheStale) {
      return cachedValue;
    }
    
    if (objWeight == 0) {
      return 0.0;
    }
    
    Arrays.fill(cachedGradient,0);

    int numRefDist = constraints.size();
    int numFeatures = trainingList.getDataAlphabet().size() + 1;
    int numLabels = trainingList.getTargetAlphabet().size();
    double scalingFactor = objWeight;      
    
    if (mapping == null) {
      // mapping maps between feature indices to 
      // constraint indices
      setMapping();
    }
    
    double[][] modelExpectations = new double[numRefDist][numLabels];
    double[][] ratio = new double[numRefDist][numLabels];
    double[] featureCounts = new double[numRefDist];

    double[][] scores = new double[trainingList.size()][numLabels];
    
    // pass 1: calculate model distribution
    for (int ii = 0; ii < trainingList.size(); ii++) {
      Instance instance = trainingList.get(ii);
      double instanceWeight = trainingList.getInstanceWeight(instance);
      
      // skip if labeled
      if (instance.getTarget() != null) {
        continue;
      }
      
      FeatureVector fv = (FeatureVector) instance.getData();
      classifier.getClassificationScoresWithTemperature(instance, temperature, scores[ii]);
      
      for (int loc = 0; loc < fv.numLocations(); loc++) {
        int featureIndex = fv.indexAtLocation(loc);
        if (constraints.containsKey(featureIndex)) {
          int cIndex = mapping.get(featureIndex);            
          double val;
          if (!useValues) {
            val = 1.;
          }
          else {
            val = fv.valueAtLocation(loc);
          }
          featureCounts[cIndex] += val;
          for (int l = 0; l < numLabels; l++) {
            modelExpectations[cIndex][l] += scores[ii][l] * val * instanceWeight;
          }
        }
      }
      
      // special case of label regularization
      if (constraints.containsKey(defaultFeatureIndex)) {
        int cIndex = mapping.get(defaultFeatureIndex); 
        featureCounts[cIndex] += 1;
        for (int l = 0; l < numLabels; l++) {
          modelExpectations[cIndex][l] += scores[ii][l] * instanceWeight;
        }        
      }
    }
    
    double value = 0;
    for (int featureIndex : constraints.keySet()) {
      int cIndex = mapping.get(featureIndex);
      if (featureCounts[cIndex] > 0) {
        for (int label = 0; label < numLabels; label++) {
          double cProb = constraints.get(featureIndex)[label];
          // normalize by count
          modelExpectations[cIndex][label] /= featureCounts[cIndex];
          ratio[cIndex][label] =  cProb / modelExpectations[cIndex][label];
          // add to the cross entropy term
          value += scalingFactor * cProb * Math.log(modelExpectations[cIndex][label]);
          // add to the entropy term
          if (cProb > 0) {
            value -= scalingFactor * cProb * Math.log(cProb);
          }
        }
        assert(Maths.almostEquals(MatrixOps.sum(modelExpectations[cIndex]),1));
      }
    }

    // pass 2: determine per example gradient
    for (int ii = 0; ii < trainingList.size(); ii++) {
      Instance instance = trainingList.get(ii);
      
      // skip if labeled
      if (instance.getTarget() != null) {
        continue;
      }
      
      double instanceWeight = trainingList.getInstanceWeight(instance);
      FeatureVector fv = (FeatureVector) instance.getData();

      for (int loc = 0; loc < fv.numLocations() + 1; loc++) {
        int featureIndex;
        if (loc == fv.numLocations()) {
          featureIndex = defaultFeatureIndex;
        }
        else {
          featureIndex = fv.indexAtLocation(loc);
        }
        
        if (constraints.containsKey(featureIndex)) {
          int cIndex = mapping.get(featureIndex);

          // skip if this feature never occurred
          if (featureCounts[cIndex] == 0) {
            continue;
          }

          double val;
          if ((featureIndex == defaultFeatureIndex)||(!useValues)) {
            val = 1;
          }
          else {
            val = fv.valueAtLocation(loc);
          }
          
          // compute \sum_y p(y|x) \hat{g}_y / \bar{g}_y
          double instanceExpectation = 0;
          for (int label = 0; label < numLabels; label++) {
            instanceExpectation += ratio[cIndex][label] * scores[ii][label];
          }
          
          // define C = \sum_y p(y|x) g_y(y,x) \hat{g}_y / \bar{g}_y
          // compute \sum_y  p(y|x) g_y(x,y) f(x,y) * (\hat{g}_y / \bar{g}_y - C)
          for (int label = 0; label < numLabels; label++) {
            if (scores[ii][label] == 0)
              continue;
            assert (!Double.isInfinite(scores[ii][label]));
            double weight = scalingFactor * instanceWeight * temperature * (val / featureCounts[cIndex]) * scores[ii][label] * (ratio[cIndex][label] - instanceExpectation);

            MatrixOps.rowPlusEquals(cachedGradient, numFeatures, label, fv, weight);
            cachedGradient[numFeatures * label + defaultFeatureIndex] += weight;
          }  
        }
      }
    }

    cachedValue = value;
    cacheStale = false;
    
    double reg = getRegularization();
    progressLogger.info ("Value (GE=" + value + " Gaussian prior= " + reg + ") = " + cachedValue);
    
    return value;
  }

  private double getRegularization() {
    double regularization;
    if (!Double.isInfinite(gaussianPriorVariance)) {
      regularization = Math.log(gaussianPriorVariance * Math.sqrt(2 * Math.PI));
    }
    else {
      regularization = 0;
    }
    for (int pi = 0; pi < parameters.length; pi++) {
      double p = parameters[pi];
      regularization -= p * p / (2 * gaussianPriorVariance);
      cachedGradient[pi] -= p / gaussianPriorVariance;
    }
    cachedValue += regularization;
    return regularization;
  }
  
  public void getValueGradient(double[] buffer) {
    if (cacheStale) {
      getValue();  
    }
    assert(buffer.length == cachedGradient.length);
    for (int i = 0; i < buffer.length; i++) {
      buffer[i] = cachedGradient[i];
    }
  }

  public int getNumParameters() {
    return parameters.length;
  }

  public double getParameter(int index) {
    return parameters[index];
  }

  public void getParameters(double[] buffer) {
    assert(buffer.length == parameters.length);
    System.arraycopy (parameters, 0, buffer, 0, buffer.length);
  }

  public void setParameter(int index, double value) {
    cacheStale = true;
    parameters[index] = value;
  }

  public void setParameters(double[] params) {
    assert(params.length == parameters.length);
    cacheStale = true;
    System.arraycopy (params, 0, parameters, 0, parameters.length);
  }
  
  private void setMapping() {
    int cCounter = 0;
    mapping = new HashMap<Integer,Integer>();
    Iterator<Integer> keys = constraints.keySet().iterator();
    while (keys.hasNext()) {
      int featureIndex = keys.next();
      mapping.put(featureIndex, cCounter);
      cCounter++;
    }
  }
}