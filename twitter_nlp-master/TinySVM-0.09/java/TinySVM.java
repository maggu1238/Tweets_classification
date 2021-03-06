/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version: 1.3.6u-20010826-1259
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */


public class TinySVM  {
  public final static native int BaseExample_add(long jarg0, String jarg1);
  public final static native int BaseExample_set(long jarg0, int jarg1, String jarg2);
  public final static native String BaseExample_get(long jarg0, int jarg1);
  public final static native int BaseExample_remove(long jarg0, int jarg1);
  public final static native int BaseExample_clear(long jarg0);
  public final static native int BaseExample_size(long jarg0);
  public final static native int BaseExample_read(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native int BaseExample_write(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native int BaseExample_readSVindex(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native int BaseExample_writeSVindex(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native void delete_BaseExample(long jarg0);
  public final static native int BaseExample_append(long jarg0, String jarg1);
  public final static native int BaseExample_appendSVindex(long jarg0, String jarg1);
  public final static native int BaseExample_getDimension(long jarg0);
  public final static native int BaseExample_getNonzeroDimension(long jarg0);
  public final static native double BaseExample_getY(long jarg0, int jarg1);
  public final static native String BaseExample_getX(long jarg0, int jarg1);
  public final static native double BaseExample_getAlpha(long jarg0, int jarg1);
  public final static native double BaseExample_getGradient(long jarg0, int jarg1);
  public final static native double BaseExample_getG(long jarg0, int jarg1);
  public final static native int Model_read(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native int Model_write(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native int Model_clear(long jarg0);
  public final static native double Model_classify(long jarg0, String jarg1);
  public final static native double Model_estimateMargin(long jarg0);
  public final static native double Model_estimateSphere(long jarg0);
  public final static native double Model_estimateVC(long jarg0);
  public final static native double Model_estimateXA(long jarg0, double jarg1);
  public final static native int Model_compress(long jarg0);
  public final static native int Model_getSVnum(long jarg0);
  public final static native int Model_getBSVnum(long jarg0);
  public final static native int Model_getTrainingDataSize(long jarg0);
  public final static native double Model_getLoss(long jarg0);
  public final static native long new_Model();
  public final static native void delete_Model(long jarg0);
  public final static native int Example_read(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native int Example_write(long jarg0, String jarg1, String jarg2, int jarg3);
  public final static native int Example_rebuildSVindex(long jarg0, long jarg1);
  public final static native long Example_learn(long jarg0, String jarg1);
  public final static native long new_Example();
  public final static native void delete_Example(long jarg0);
}
