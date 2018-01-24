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
open System.Reactive.Concurrency

let retrieveFrame channel (capture:VideoCapture) =
    let frame = new Mat()
    (capture.Retrieve(frame, channel),frame)

let tryDetectFace (frame:Mat) =
    let grayscaled = frame |> grayScale

    grayscaled |> 
        equalizeHistogram |> 
        tryExtractFace |> 
        (fun maybeFace -> maybeFace, frame, grayscaled)

let getFeatures ((_, eyes), frame, grayscaled) = 
        
    eyes |> List.toArray |> 
        Array.collect (goodFeaturesToTrack grayscaled) |> 
        (fun features -> frame, features)

let imageGrabbedObservable (capture:VideoCapture) = 
    capture.ImageGrabbed |> 
                Observable.map (fun _ -> capture) |> 
                Observable.filter (fun cap -> cap.Ptr <> IntPtr.Zero) |> 
                Observable.map (retrieveFrame 0) |>
                Observable.filter fst |> 
                Observable.map snd

let useFrame status error = 
    let totalMissingFeatures = status |> Array.filter ((=)(Convert.ToByte 0)) |> Array.length
    let totalError = error |> Array.sum
    not (totalError > 75.0f || totalMissingFeatures > 0)        

let trackFeatures (previousFrame, previousFeatures) (currentFrame, currentFeatures) =
    match currentFeatures |> Array.length with
        | 0 -> let features, _, _ = lucasKanade (grayScale currentFrame) (grayScale previousFrame) previousFeatures 
               in currentFrame, features
        
        | _ -> let features, status, error = lucasKanade (grayScale previousFrame) (grayScale currentFrame) currentFeatures 
               in if useFrame status error then previousFrame, features else previousFrame, previousFeatures

[<EntryPoint>]
let main _ =
    let capture = new VideoCapture()
    capture.Start()

    let imageGrabbed = capture |> imageGrabbedObservable
    
    let eventLoopScheduler = new EventLoopScheduler()

    let faceDetectionObservable = 
        imageGrabbed |> 
            Observable.bufferCount 15 |>
            Observable.map Seq.last |>
            Observable.observeOn eventLoopScheduler |>
            Observable.map tryDetectFace |>
            Observable.filter (fun (maybeFace,_,_) -> Option.isSome <| maybeFace) |>
            Observable.map (fun (maybeFace, frame, gray) -> 
                                 getFeatures (Option.get maybeFace, frame, gray))
                                                                              
    imageGrabbed |>
        Observable.map (fun frame -> frame, Array.empty) |>
        Observable.merge faceDetectionObservable |>
        Observable.scan trackFeatures |>
        Observable.subscribe (fun (frame, points) -> 
            // a bounding box that contains all points should contain the eyes
            let output = new Mat();
            let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
            Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
            CvInvoke.Imshow("Dojutsify Me", output)
            CvInvoke.WaitKey(1) |> ignore) |> ignore

    Console.ReadKey() |> ignore    
    0