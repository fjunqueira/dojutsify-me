module DojutsifyMe.FaceTracking

open Emgu.CV;
open Emgu.CV.Features2D;
open Emgu.CV.Structure;
open System.Drawing;
open Emgu.CV.Util;

let goodFeaturesToTrack (face:Mat) = 
    let keyPointDetector = new GFTTDetector() //1000, 0.01, 1.0, 3, false, 0.04
    let modelDescriptors = new Mat();
    let modelKeyPoints = new VectorOfKeyPoint();
    keyPointDetector.DetectAndCompute(face, null, modelKeyPoints, modelDescriptors ,false)
    ()