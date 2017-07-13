module DojutsifyMe.Main

open System;
open System.Drawing;
open System.Windows.Forms;
open Emgu.CV;
open Emgu.CV.UI;
open Emgu.CV.CvEnum;
open Emgu.CV.Structure;
open Emgu.CV.Features2D;
open DojutsifyMe.FaceDetection;
open DojutsifyMe.FaceTracking;
open DojutsifyMe.ImageProcessing;
open FSharp.Control;
open FSharp.Control.Reactive;
open FSharp.Control.Reactive.Observable;
open FSharpx.Choice
open Emgu.CV.Util;
open FSharpx

let mainBox = new ImageBox(Location=Point(0,20), Size=Size(500,500), Image=null)
let secondBox = new ImageBox(Location=Point(500,0), Size=Size(300,250), Image=null)
let thirdBox = new ImageBox(Location=Point(500,250), Size=Size(300,250), Image=null)
let message = new Label(Location=Point(0,0),Size=Size(500,13))

let display (imageBox:ImageBox) (image:Mat) = 
    imageBox.Image <- image

let retrieveFrame channel (capture:VideoCapture) =
    let frame = new Mat()
    (capture.Retrieve(frame, channel),frame)

let drawRectangle color (frame:Mat) rectangle =
    CvInvoke.Rectangle(frame, rectangle, Bgr(color).MCvScalar, 2)

let tryDetectFace (frame:Mat) =
    let grayscaled = frame |> grayScale

    grayscaled |> 
        equalizeHistogram |> 
        tryExtractFace |> 
        (fun maybeFace -> maybeFace, frame, grayscaled)

let getFeatures (face, frame, grayscaled) = 
        
    face |> 
        snd |> 
        List.toArray |> 
        Array.collect (goodFeaturesToTrack grayscaled) |> 
        (fun features -> frame, features) |>
        (fun ((frame, features) as data) -> 
            let output = new Mat();
            let keypoints = new VectorOfKeyPoint(features |> Array.map (fun p -> MKeyPoint(Point=p)))
            Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
            CvInvoke.Resize(output, output, Size(300, 150), 0.0, 0.0, Inter.Linear)
            secondBox.Image <- output
            data)

let imageGrabbedObservable (capture:VideoCapture) = 
    capture.ImageGrabbed |> 
                Observable.map (fun _ -> capture) |> 
                Observable.filter (fun cap -> cap.Ptr <> IntPtr.Zero) |> 
                Observable.map (retrieveFrame 0) |>
                Observable.filter fst |> 
                Observable.map snd

type FaceDetectionStatus = Detected | NotDetected | NoDetectionAttempted

let trackFeatures (previousStatus, previousFrame, previousFeatures, _, _) (currentStatus, currentFrame, currentFeatures, _, _) =
    match currentStatus with
        | NotDetected | NoDetectionAttempted -> let features, status, trackError = lucasKanade (grayScale currentFrame) (grayScale previousFrame) previousFeatures 
                                                in currentStatus, currentFrame, features, status, trackError

        | _ -> currentStatus, currentFrame, currentFeatures, Array.empty, Array.empty

[<EntryPoint>]
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")
    
    form.Controls.AddRange([|mainBox;secondBox;thirdBox;message|])

    let capture = new VideoCapture()
    capture.Start()

    let imageGrabbed = capture |> imageGrabbedObservable

    let faceDetectionTrigger = new Subject<unit>()

    let faceDetectionObservable = 
        faceDetectionTrigger |> 
            Observable.throttle (TimeSpan.FromMilliseconds 20.0) |>
            Observable.map (fun data -> printfn "%s" "Started faceDetectionTrigger"; data) |>
            Observable.flatmap (fun _ -> imageGrabbed |> 
                                            Observable.first |>
                                            Observable.map tryDetectFace |>  
                                            Observable.map (fun (maybeFace, frame, gray) -> 
                                                                 match maybeFace with
                                                                  | None -> NotDetected, frame, Array.empty, Array.empty, Array.empty
                                                                  | Some face -> getFeatures (face, frame, gray) |> 
                                                                                     (fun (frame, features) -> Detected, frame, features, Array.empty, Array.empty))) |>
            Observable.map (fun data -> printfn "%s" "Ended faceDetectionTrigger"; data)
                                                                                     
    use webcamImageProcessor =                                                                       
        imageGrabbed |>
            Observable.map (fun frame -> NoDetectionAttempted, frame, Array.empty, Array.empty, Array.empty) |>
            Observable.merge faceDetectionObservable |>
            Observable.map (fun ((status,_,_,_,_) as data) -> match status with
                                                                 | Detected | NotDetected -> printfn "%s" "Got the frame from faceDetectionTrigger"
                                                                 | _ -> printfn "%s" "Got the frame from webcamImageProcessor"; 
                                                              data) |>
            Observable.scan trackFeatures |>
            Observable.subscribe (fun (_, frame, points, status, trackError) -> 

                let totalMissingFeatures = status |> Array.filter ((=)(Convert.ToByte 0)) |> Array.length
                let totalError = trackError |> Array.sum

                if totalError > 75.0f || totalMissingFeatures > 0 then faceDetectionTrigger.OnNext(()) else () |> ignore
                
                let output = new Mat();
                let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
                Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
                CvInvoke.Resize(output, output, Size(500, 300), 0.0, 0.0, Inter.Linear)
                mainBox.Image <- output)

    use detectionStatusObservable = 
        faceDetectionObservable |>
            Observable.map (fun (status,_,_,_,_) -> status) |>
            Observable.subscribe (function NotDetected -> message.Text <- "Cannot detect face" | _ -> message.Text <- String.Empty)

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0