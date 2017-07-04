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
open FSharp.Control.Reactive;
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

let tryFaceDetectedObservable (frame:Mat) =
    let grayscaled = frame |> grayScale
    let equalized = grayscaled |> equalizeHistogram
    equalized |> extractFace |> Observable.single |> Observable.map (fun data -> data, frame, grayscaled)

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

let featuresDetectedObservable grabber = 
    grabber |> 
        Observable.flatmap faceDetectedObservable |> 
        Observable.first |> 
        Observable.flatmap imageFeaturesObservable



let faceTrackingObservable initialFrame initialPoints grabber = 
    // Prevent multiple calls to featuresDetectedObservable from getting accumulated when we can't detect a face
    // every 5 secs a new observable will be created, the previous one must be finished by then

//will have to recreate it everytime
    let interval = Observable.interval (TimeSpan.FromSeconds 5.0) |> Observable.flatmap (fun _ -> grabber |> 
                                                                                                        Observable.flatmap tryFaceDetectedObservable |> 
                                                                                                        Observable.first |> 
                                                                                                        Observable.flatmap (fun ((detected, data), frame, gray) -> match detected with 
                                                                                                                                                                   | true -> imageFeaturesObservable (data, frame, gray)
                                                                                                                                                                   | false -> Observable.single (frame, Array.empty))) 

    grabber |>
        Observable.map (fun frame -> frame, Array.empty) |>
        Observable.merge interval |>
        Observable.scanInit 
            (initialFrame, initialPoints) 
            (fun (previousFrame, previousFeatures) ((nextFrame, newFeatures) as next) -> 
                match newFeatures.Length with
                    | 0 -> let currentPoints, status, _ = lucasKanade (grayScale nextFrame) (grayScale previousFrame) previousFeatures in nextFrame, currentPoints
                    | _ -> next)

[<EntryPoint>]
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")
    
    form.Controls.AddRange([|mainBox;secondBox;thirdBox|])

    let capture = new VideoCapture()
    capture.Start()
    
    let imageGrabbed = capture |> imageGrabbedObservable
           
    use processFrame = 
            featuresDetectedObservable imageGrabbed |> 
            Observable.flatmap (fun (frame, features) -> faceTrackingObservable frame features imageGrabbed) |>
            Observable.subscribe (fun (frame, points) -> 
              
                // let totalMissingFeatures = (status |> Array.filter ((=)(Convert.ToByte 0)) |> Array.length)
                // let totalError = (trackError |> Array.sum)

                // if totalMissingFeatures > 0 then printfn "%d features were not found" totalMissingFeatures else () |> ignore
                // if totalError > 26.0f then printfn "%f total error" totalError else () |> ignore
    
                // depending on the error level draw error frame and wait for refresh

                let output = new Mat();
                let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
                Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
                CvInvoke.Resize(output, output, Size(500, 300), 0.0, 0.0, Inter.Linear)
                mainBox.Image <- output)

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0