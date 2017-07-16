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
open System.Reactive.Concurrency
open System.Reactive
open System.Threading

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
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")
    
    form.Controls.AddRange([|mainBox;secondBox;thirdBox;message|])

    let capture = new VideoCapture()
    capture.Start()

    let imageGrabbed = capture |> imageGrabbedObservable
    
    let eventLoopScheduler = new EventLoopScheduler()

    let controlScheduler = ControlScheduler(form)

    let faceDetectionObservable = 
        imageGrabbed |> 
            Observable.bufferCount 15 |>
            Observable.map Seq.last |>
            Observable.observeOn eventLoopScheduler |>
            Observable.map tryDetectFace |>
            Observable.filter (fun (maybeFace,_,_) -> Option.isSome <| maybeFace) |>
            Observable.map (fun (maybeFace, frame, gray) -> 
                                 getFeatures (Option.get maybeFace, frame, gray))
            
    use webcamImageProcessor =                                                                       
        imageGrabbed |>
            Observable.map (fun frame -> frame, Array.empty) |>
            Observable.merge faceDetectionObservable |>
            Observable.scan trackFeatures |>
            Observable.observeOn controlScheduler |>    
            Observable.subscribe (fun (frame, points) -> 
                let output = new Mat();
                let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
                Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
                CvInvoke.Resize(output, output, Size(500, 300), 0.0, 0.0, Inter.Linear)

                mainBox.Image <- output)

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0