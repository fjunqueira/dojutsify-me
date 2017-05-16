module DojutsifyMe.FaceTracking

open Emgu.CV;
open Emgu.CV.Features2D;
open Emgu.CV.Structure;
open System.Drawing;
open Emgu.CV.Util;
open System.Xml;
open Emgu.CV;
open Emgu.CV.CvEnum;
open Emgu.CV.Features2D;
open Emgu.CV.Structure;
open Emgu.CV.Util;
open Emgu.CV.XFeatures2D;
open Emgu.Util;

let goodFeaturesToTrack (frame:Mat) face = 
    let keyPointDetector = new GFTTDetector()

    let mask = new Mat(frame.Size, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
    mask.SetTo(MCvScalar(0.0))

    let roi = new Mat(mask, face);
    roi.SetTo(MCvScalar(255.0))
    
    let modelKeyPoints = new VectorOfKeyPoint();
    keyPointDetector.DetectRaw(frame, modelKeyPoints, mask)
    modelKeyPoints    