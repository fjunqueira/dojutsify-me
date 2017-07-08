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

let imageGrabbedObservable (capture:VideoCapture) = 
    capture.ImageGrabbed |> 
                Observable.map (fun _ -> capture) |> 
                Observable.filter (fun cap -> cap.Ptr <> IntPtr.Zero) |> 
                Observable.map (retrieveFrame 0) |>
                Observable.filter fst |> 
                Observable.map snd

let faceDetectedObservable (frame:Mat) =
    let grayscaled = frame |> grayScale
    let equalized = grayscaled |> equalizeHistogram
    
    equalized |>        
        extractFace |> 
        Observable.single |> 
        Observable.filter fst |>
        Observable.map (fun data -> (data |> snd), frame, grayscaled)

let maybeFaceDetectedObservable (frame:Mat) =
    let grayscaled = frame |> grayScale
    let equalized = grayscaled |> equalizeHistogram

    equalized |> 
        extractFace |> 
        Observable.single |> 
        Observable.map (fun data -> (if fst data then Some (snd data) else None), frame, grayscaled)

let imageFeaturesObservable (face, frame, grayscaled) = 
        
       face |>
        Observable.single |>
        Observable.map (fun (_, eyes) -> frame, eyes |> (List.toArray >> Array.collect (goodFeaturesToTrack grayscaled))) |>
        Observable.map (fun ((frame, points) as data) -> 
            let output = new Mat();
            let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
            Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
            CvInvoke.Resize(output, output, Size(300, 150), 0.0, 0.0, Inter.Linear)
            secondBox.Image <- output
            data)

let faceTrackingObservable initialFrame initialPoints grabber redetectFace = 

    let interval = redetectFace |> 
                   Observable.throttle (TimeSpan.FromMilliseconds 38.0) |>
                   Observable.flatmap (fun _ -> grabber |> 
                                                   Observable.first |>
                                                   Observable.flatmap maybeFaceDetectedObservable |>  
                                                   Observable.flatmap (fun (data, frame, gray) -> match data with 
                                                                                                  | Some face -> imageFeaturesObservable (face, frame, gray) |> 
                                                                                                                    Observable.map (fun (a,b) -> a, b, Array.empty, Array.empty)
                                                                                                  // place the "put your face in front of the camera" message here
                                                                                                  | None -> Observable.single (frame, Array.empty, Array.empty, Array.empty)))

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
    
    use processFrame = 
          imageGrabbed |>
            Observable.flatmap faceDetectedObservable |> 
            // place the "put your face in front of the camera" message here in another subscriber
            Observable.first |>
            Observable.flatmap imageFeaturesObservable |> 
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