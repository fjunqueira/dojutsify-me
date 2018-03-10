module DojutsifyMe.ImageProcessing

open Emgu.CV
open Emgu.CV.Structure
open Emgu.CV.CvEnum
open Emgu.CV.Util

type GrayScaled = GrayScaled of Mat
type EqualizedHistogram = EqualizedHistogram of Mat

let grayScale image = 
    let grayImage = new Mat()
    CvInvoke.CvtColor(image, grayImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray)
    GrayScaled grayImage

let equalizeHistogram (GrayScaled image) = 
    let equalizedImage = new Mat()
    CvInvoke.EqualizeHist(image, equalizedImage)
    EqualizedHistogram equalizedImage

let rangeTreshold minValue maxValue (image:Mat) = 
    let tresholded = new Mat()
    CvInvoke.InRange(image, new ScalarArray(new MCvScalar(minValue)), new ScalarArray(new MCvScalar(maxValue)), tresholded)
    tresholded

let canny image = 
    let edges = new Mat()
    CvInvoke.Canny(image, edges, 75.0, 200.0)
    edges

let findContours image = 
    let contours = new VectorOfVectorOfPoint();
    CvInvoke.FindContours(image, contours, null, RetrType.Tree, ChainApproxMethod.ChainApproxSimple)
    contours.ToArrayOfArray()