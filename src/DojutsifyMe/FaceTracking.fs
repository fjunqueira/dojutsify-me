module DojutsifyMe.FaceTracking

open System;
open Emgu.CV;
open Emgu.CV.Features2D;
open Emgu.CV.Structure;
open System.Drawing;
open Emgu.CV.Util;
open Emgu.CV;
open Emgu.CV.CvEnum;
open Emgu.CV.Features2D;
open Emgu.CV.Structure;
open Emgu.CV.Util;
open Emgu.CV.XFeatures2D;
open Emgu.Util;
open DojutsifyMe.ImageProcessing;

let goodFeaturesToTrack (GrayScaled frame) face = 
    let keyPointDetector = new GFTTDetector(5, 0.01, 1.0, 3, false, 0.04);

    let mask = new Mat(frame.Size, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
    mask.SetTo(MCvScalar(0.0))

    let roi = new Mat(mask, face);
    roi.SetTo(MCvScalar(255.0))
    
    keyPointDetector.Detect(frame, mask) |> Array.map (fun p -> p.Point)

let lucasKanade (GrayScaled nextFrame) (GrayScaled previousFrame) (previousPoints:PointF[]) =
   
    let mutable currentPoints = Unchecked.defaultof<PointF[]>
    let mutable status = Unchecked.defaultof<byte[]>
    let mutable trackError = Unchecked.defaultof<float32[]>

    // this is an ugly workaround to prevent an exception from ocurring in the CalcOpticalFlowPyrLK when the previous points are empty
    // and outputing a result that makes the main process try to find a feature in the following frames
    // i'll think of a better way to do this later
    if previousPoints.Length = 0 then (Array.zeroCreate 0, Array.create 1 (Convert.ToByte 0), Array.create 1 100.0f) else

    CvInvoke.CalcOpticalFlowPyrLK(
        previousFrame, 
        nextFrame, 
        previousPoints, 
        Size(15,15), 
        2, 
        MCvTermCriteria(10, 0.03), 
        &currentPoints, 
        &status, 
        &trackError)

    (currentPoints, status, trackError)