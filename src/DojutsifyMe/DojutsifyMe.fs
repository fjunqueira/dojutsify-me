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
open FSharp.Control.Reactive;
open FSharpx.Choice
open Emgu.CV.Util;
open FSharpx

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
    frame |> 
        extractFace |> 
        Observable.single |> 
        Observable.filter fst |>
        Observable.map snd |>
        Observable.map (fun (head,_) -> head, frame, goodFeaturesToTrack frame head)

[<EntryPoint>]
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")

    let mainBox = new ImageBox(Location=Point(0,0), Size=Size(500,500), Image=null)
    let secondBox = new ImageBox(Location=Point(500,0), Size=Size(300,250), Image=null)
    let thirdBox = new ImageBox(Location=Point(500,250), Size=Size(300,250), Image=null)
    
    form.Controls.AddRange([|mainBox;secondBox;thirdBox|])

    let capture = new VideoCapture()
    capture.Start()
    
    use processFrame = 
            capture |>
                imageGrabbedObservable |> 
                Observable.flatmap imageFeaturesObservable |>
                Observable.subscribe 
                    (fun (head, frame, goodFeatures) -> 
                        let output = new Mat();
                        Features2DToolbox.DrawKeypoints (frame, goodFeatures, output, Bgr(Color.Green), Features2DToolbox.KeypointDrawType.Default)
                        mainBox.Image <- output)
                //Observable.first |>
                //Observable.flatmap (fun data -> capture |> imageGrabbedObservable |> Observable.map (data |> snd |> tuple2)) |>
                //Observable.subscribe (fun (face, frame) -> mainBox.Image <- frame)

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0