module DojutsifyMe.Main

open System;
open System.Drawing;
open Emgu.CV;
open Emgu.CV.Structure;
open Emgu.CV.Features2D;
open DojutsifyMe.FaceDetection;
open DojutsifyMe.FaceTracking;
open DojutsifyMe.ImageProcessing;
open FSharp.Control;
open FSharp.Control.Reactive;
open Emgu.CV.Util;
open FSharpx
open FSharpx.Reader

type PipelineArguments = {
    frame : Mat; 
    gray : GrayScaled; 
    equalized : EqualizedHistogram; 
    previousGray : GrayScaled; 
    previousPoints : PointF list }

let retrieveFrame channel (capture:VideoCapture) =
    let frame = new Mat()
    (capture.Retrieve(frame, channel), frame)

let tryFindingFace = Reader.asks (fun args -> args.equalized |> tryFindingFace)

let tryFindingEyes maybeFace = Reader.asks (fun args -> Option.map (tryFindingEyes args.equalized) maybeFace |> Option.flatten)

let trackFeatures = Reader.asks (fun args -> lucasKanade args.gray args.previousGray args.previousPoints)

let getFeatures eyes = Reader.asks (fun args -> List.collect (goodFeaturesToTrack args.gray) eyes)

let frameProcessingPipeline = 
    reader {
        let! args = ask

        let! maybeFace = tryFindingFace
        let! maybeEyes = tryFindingEyes maybeFace
        
        if Option.isSome maybeEyes then
            let! features = getFeatures <| Option.get maybeEyes

            let (EqualizedHistogram frame) = args.equalized
            let faceDebug = frame.Clone()
            Option.get maybeEyes |> List.map (fun eye -> CvInvoke.Rectangle(faceDebug, eye, MCvScalar(255.0, 0.0, 0.0))) |> ignore
            CvInvoke.Rectangle(faceDebug, (Option.get maybeFace), MCvScalar(0.0, 0.0, 255.0))
            CvInvoke.Imshow("Eyes View", faceDebug)
            CvInvoke.WaitKey(1) |> ignore

            return features
        else            
            let! (newPoints, _, _) = trackFeatures
            return newPoints
    }

let processFrame (previousFrame, previousPoints) currentFrame =

    let grayscaled = grayScale currentFrame
    let equalized  = equalizeHistogram grayscaled

    let pipelineArguments = {
        frame = currentFrame; 
        gray = grayscaled; 
        equalized = equalized; 
        previousGray = grayScale previousFrame; 
        previousPoints = previousPoints }

    let currentPoints = frameProcessingPipeline pipelineArguments
    (currentFrame, currentPoints)

[<EntryPoint>]
let main _ =
    let capture = new VideoCapture()
    capture.Start()

    let initData = (new Mat(Size(100,100), CvEnum.DepthType.Cv32F, 3) , [])

    capture.ImageGrabbed |> 
        Observable.map (fun _ -> capture) |> 
        Observable.filter (fun cap -> cap.Ptr <> IntPtr.Zero) |> 
        Observable.map (retrieveFrame 0) |>
        Observable.filter fst |> 
        Observable.map snd |>
        Observable.scanInit initData processFrame |>
        Observable.subscribe (fun (frame, points) ->
            let output = frame.Clone()
            
            let keypoints = new VectorOfKeyPoint(points |> List.map (fun p -> MKeyPoint(Point=p)) |> List.toArray)
            Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
            
            CvInvoke.Imshow("Dojutsify Me", output)
            CvInvoke.WaitKey(1) |> ignore
        ) |> ignore

    Console.ReadKey() |> ignore    
    0