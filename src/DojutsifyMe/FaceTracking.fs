module DojutsifyMe.FaceTracking

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
    let keyPointDetector = new GFTTDetector()

    let mask = new Mat(frame.Size, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
    mask.SetTo(MCvScalar(0.0))

    let roi = new Mat(mask, face);
    roi.SetTo(MCvScalar(255.0))
    
    keyPointDetector.Detect(frame, mask)

let lucasKanade (GrayScaled nextFrame) (GrayScaled previousFrame) (previousPoints:MKeyPoint[]) =
    
    let mutable currFeatures = Unchecked.defaultof<PointF[]>
    let mutable status = Unchecked.defaultof<byte[]>
    let mutable trackError = Unchecked.defaultof<float32[]>

    CvInvoke.CalcOpticalFlowPyrLK(
        previousFrame, 
        nextFrame, 
        previousPoints |> Array.map (fun p -> p.Point), 
        Size(15,15), 
        2, 
        MCvTermCriteria(10, 0.03), 
        &currFeatures, 
        &status, 
        &trackError);
    
    printfn "Previous point count %d" (previousPoints |> Array.length);
    printfn "Current point count %d" (currFeatures |> Array.length);