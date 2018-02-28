module DojutsifyMe.Main

open System;
open System.Drawing;
open Emgu.CV;
open Emgu.CV.Structure;
open Emgu.CV.Features2D;
open DojutsifyMe.FaceDetection;
open DojutsifyMe.FaceTracking;
open DojutsifyMe.ImageProcessing;
open DojutsifyMe.Utils;
open FSharp.Control;
open FSharp.Control.Reactive;
open Emgu.CV.Util;
open FSharpx
open FSharpx.Reader
open FSharpx

// Maximum number of features that should be tracked (5 per eye)
let maxCorners = 10

type PipelineArguments = {
    frame : Mat; 
    gray : GrayScaled; 
    equalized : EqualizedHistogram; 
    previousGray : GrayScaled; }

let retrieveFrame channel (capture:VideoCapture) =
    let frame = new Mat()
    (capture.Retrieve(frame, channel), frame)

let tryFindingFace = Reader.asks (fun args -> args.gray |> tryFindingFace)

let tryFindingEyes maybeFace = Reader.asks (fun args -> Option.map (tryFindingEyes args.gray) maybeFace |> Option.flatten)

let tryFindingFeatures maybeEyes = Reader.asks (fun args -> Option.map (mapTuple <| goodFeaturesToTrack args.gray maxCorners) maybeEyes)

// let trackFeatures = 
//     let mergeTrackingResult ((lp, ls, lt), (rp, rs, rt)) = ((lp, rp), List.append ls rs, List.append lt rt)
//     Reader.asks (fun args -> mapTuple (lucasKanade args.gray args.previousGray) args.previousPoints |> mergeTrackingResult)

let frameProcessingPipeline = 
    reader {
        
        // let! (newPoints, status, trackError) = trackFeatures

        // let featureDisappeared = status |> List.exists (fun s -> s = byte 0)
        // let maximumErrorExceeded = trackError |> List.sum > 50.0f
        // let noPointsAvailable = List.isEmpty status 

        // if maximumErrorExceeded || featureDisappeared || noPointsAvailable then
        //     let! maybeFace = tryFindingFace
        //     let! maybeEyes = tryFindingEyes maybeFace
        //     let! maybeFeatures = tryFindingFeatures maybeEyes
            
        //     //let maybeMeans = Option.map (mapTuple meanOfPoints) maybeFeatures
            
        //     return Option.getOrElse newPoints maybeFeatures
        // else
        //     printfn "Max error: %s" (List.max >> string <| trackError)
        //     return newPoints

        let! maybeFace = tryFindingFace
        let! maybeEyes = tryFindingEyes maybeFace

        if Option.isSome maybeEyes then
            let! args = ask
            let faceDebug = args.frame.Clone()
            
            let face = Option.get maybeFace
            let (leftEye, rightEye) = Option.get maybeEyes

            CvInvoke.Rectangle(faceDebug, face, MCvScalar(0.0, 0.0, 255.0))
            CvInvoke.Rectangle(faceDebug, leftEye, MCvScalar(255.0, 0.0, 0.0))
            CvInvoke.Rectangle(faceDebug, rightEye, MCvScalar(255.0, 0.0, 0.0))
            CvInvoke.Imshow("Debug View", faceDebug)
            CvInvoke.WaitKey(1) |> ignore

        return ()
    }

let processFrame previousFrame currentFrame =

    let grayscaled = grayScale currentFrame
    let equalized  = equalizeHistogram grayscaled

    let pipelineArguments = {
        frame = currentFrame; 
        gray = grayscaled; 
        equalized = equalized; 
        previousGray = grayScale previousFrame; }

    frameProcessingPipeline pipelineArguments |> ignore
    currentFrame

[<EntryPoint>]
let main _ =
    let capture = new VideoCapture()
    capture.Start()

    let initData = new Mat(Size(100,100), CvEnum.DepthType.Cv32F, 3)

    capture.ImageGrabbed |> 
        Observable.map (fun _ -> capture) |> 
        Observable.filter (fun cap -> cap.Ptr <> IntPtr.Zero) |> 
        Observable.map (retrieveFrame 0) |>
        Observable.filter fst |> 
        Observable.map snd |>
        Observable.scanInit initData processFrame |>
        Observable.subscribe (fun frame ->
            //let points = List.append leftPoints rightPoints
            
            let output = frame.Clone()
            
            //let keypoints = new VectorOfKeyPoint(points |> List.map (fun p -> MKeyPoint(Point=p)) |> List.toArray)
            //Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
            
            CvInvoke.Imshow("Dojutsify Me", output)
            CvInvoke.WaitKey(1) |> ignore
        ) |> ignore

    Console.ReadKey() |> ignore    
    0