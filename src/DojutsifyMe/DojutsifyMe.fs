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

let imageFeaturesObservable frame = 
    let grayscaled = frame |> grayScale
    let equalized = grayscaled |> equalizeHistogram
    
    equalized |>        
        extractFace |> 
        Observable.single |> 
        Observable.filter fst |>
        Observable.map snd |>
        Observable.map (fun (head, _) -> frame, goodFeaturesToTrack grayscaled head) |>
        Observable.map (fun ((frame, points) as data) -> 
            let output = new Mat();
            let keypoints = new VectorOfKeyPoint(points |> Array.map (fun p -> MKeyPoint(Point=p)))
            Features2DToolbox.DrawKeypoints(frame, keypoints, output, Bgr(Color.Green),Features2DToolbox.KeypointDrawType.Default)
            CvInvoke.Resize(output, output, Size(300, 150), 0.0, 0.0, Inter.Linear)
            secondBox.Image <- output
            data)

let faceTrackingObservable initialFrame initialPoints capture =
    capture |> 
        imageGrabbedObservable |>
        Observable.scanInit 
            (initialFrame, initialPoints) 
            (fun previous next -> let currentPoints, status, _ = lucasKanade (grayScale next) (previous |> (fst >> grayScale)) (previous |> snd) in next, currentPoints)
        
[<EntryPoint>]
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")
    
    form.Controls.AddRange([|mainBox;secondBox;thirdBox|])

    let capture = new VideoCapture()
    capture.Start()
    
    use processFrame = 
            capture |>
                imageGrabbedObservable |> 
                Observable.flatmap imageFeaturesObservable |>
                Observable.first |>
                Observable.flatmap (fun (frame, features) -> faceTrackingObservable frame features capture) |>
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