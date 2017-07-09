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

let mainBox = new ImageBox(Location=Point(0,0), Size=Size(500,500), Image=null)
let secondBox = new ImageBox(Location=Point(500,0), Size=Size(300,250), Image=null)
let thirdBox = new ImageBox(Location=Point(500,250), Size=Size(300,250), Image=null)

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
        (fun eyes -> frame, eyes) |>
        (fun ((frame, points) as data) -> 
            let output = new Mat();
            let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
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

let faceTrackingObservable initialFrame initialPoints grabber redetectFace = 

    let interval = redetectFace |> 
                   Observable.throttle (TimeSpan.FromMilliseconds 38.0) |>
                   Observable.flatmap (fun _ -> grabber |> 
                                                   Observable.first |>
                                                   Observable.map tryDetectFace |>  
                                                   Observable.flatmap (fun (data, frame, gray) -> 
                                                                        match data with
                                                                         // place the "put your face in front of the camera" message here
                                                                         | None -> Observable.single (frame, Array.empty, Array.empty, Array.empty)
                                                                         | Some face -> getFeatures (face, frame, gray) |> 
                                                                                            (fun (frame, eyes) -> frame, eyes, Array.empty, Array.empty) |>
                                                                                            Observable.single))

    grabber |>
        Observable.map (fun frame -> frame, Array.empty, Array.empty, Array.empty) |>
        Observable.merge interval |>
        Observable.scanInit 
            (initialFrame, initialPoints, Array.empty, Array.empty) 
            (fun (previousFrame, previousFeatures, _, _) ((nextFrame, newFeatures, _, _) as next) -> 
                match newFeatures.Length with
                    | 0 -> let currentPoints, status, trackError = lucasKanade (grayScale nextFrame) (grayScale previousFrame) previousFeatures in nextFrame, currentPoints, status, trackError
                    | _ -> next)

[<EntryPoint>]
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")
    
    form.Controls.AddRange([|mainBox;secondBox;thirdBox|])

    let capture = new VideoCapture()
    capture.Start()
    
    let imageGrabbed = capture |> imageGrabbedObservable

    let redetectFace = new Subject<unit>()
    
    let maybeFaceDetectedObservable = 
        imageGrabbed |> 
        Observable.throttle (TimeSpan.FromMilliseconds 38.0) |> 
        Observable.map tryDetectFace            

    // place the "put your face in front of the camera" message here in another subscriber
    use faceNotDetected = 
        maybeFaceDetectedObservable |>
            Observable.takeUntilOther (maybeFaceDetectedObservable |> Observable.filter (fun (data, _, _) -> data |> Option.isSome)) |>
            Observable.filter (fun (data, _, _) -> data |> Option.isNone) |>
            Observable.map (fun (_, frame, _) -> frame) |>
            Observable.subscribe (fun frame -> mainBox.Image <- frame)

    use processFrame = 
          maybeFaceDetectedObservable |>
            Observable.filter (fun (data, _, _) -> data |> Option.isSome) |>
            Observable.map (fun (data, frame, grayscaled) -> (data |> Option.get), frame, grayscaled) |>
            Observable.map getFeatures |> 
            Observable.first |>
            Observable.flatmap (fun (frame, features) -> faceTrackingObservable frame features imageGrabbed redetectFace) |>
            Observable.subscribe (fun (frame, points, status, trackError) -> 

                let totalMissingFeatures = status |> Array.filter ((=)(Convert.ToByte 0)) |> Array.length
                let totalError = trackError |> Array.sum

                if totalError > 75.0f || totalMissingFeatures > 0 then redetectFace.OnNext(()) else () |> ignore
    
                let output = new Mat();
                let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
                Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
                CvInvoke.Resize(output, output, Size(500, 300), 0.0, 0.0, Inter.Linear)
                mainBox.Image <- output)

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0