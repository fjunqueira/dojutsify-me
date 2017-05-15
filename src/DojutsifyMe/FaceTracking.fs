module DojutsifyMe.FaceTracking

open Emgu.CV;
open Emgu.CV.Features2D;
open Emgu.CV.Structure;
open System.Drawing;
open Emgu.CV.Util;

let (|Point|) (point : PointF) = ( point.X, point.Y)

let goodFeaturesToTrack (face:Mat) = 
    let keyPointDetector = new GFTTDetector()
    let modelDescriptors = new Mat();
    let modelKeyPoints = new VectorOfKeyPoint();
    keyPointDetector.DetectRaw(face, modelKeyPoints, modelDescriptors)
    modelKeyPoints
    