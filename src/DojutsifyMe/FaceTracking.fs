module DojutsifyMe.FaceTracking

open Emgu.CV;
open Emgu.CV.Features2D;
open Emgu.CV.Structure;
open System.Drawing;
open DojutsifyMe.ImageProcessing;

let goodFeaturesToTrack (GrayScaled frame) maxCorners roiArea = 
    let keyPointDetector = new GFTTDetector(maxCorners, 0.01, 1.0, 3, false, 0.04);

    let mask = new Mat(frame.Size, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
    mask.SetTo(MCvScalar(0.0))

    let roi = new Mat(mask, roiArea);
    roi.SetTo(MCvScalar(255.0))
    
    keyPointDetector.Detect(frame, mask) |> Array.toList |> List.map (fun p -> p.Point)

let lucasKanade (GrayScaled nextFrame) (GrayScaled previousFrame) (previousPoints:PointF list) =
   
    let mutable currentPoints = Unchecked.defaultof<PointF[]>
    let mutable status = Unchecked.defaultof<byte[]>
    let mutable trackError = Unchecked.defaultof<float32[]>

    // This is an ugly workaround to prevent an exception from ocurring in the CalcOpticalFlowPyrLK function when the previousPoints are empty
    // It outputs a result that makes the main process try to find a feature in the next frames
    // I'll think of a better way to do this later
    if previousPoints.Length = 0 then ([], [], []) else

    CvInvoke.CalcOpticalFlowPyrLK(
        previousFrame, 
        nextFrame, 
        previousPoints |> List.toArray,
        Size(15, 15), 
        2, 
        MCvTermCriteria(10, 0.03), 
        &currentPoints, 
        &status, 
        &trackError)

    (Array.toList currentPoints, Array.toList status, Array.toList trackError)