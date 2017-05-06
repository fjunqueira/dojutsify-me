module DojutsifyMe.Main

open System;
open System.Drawing;
open System.Windows.Forms;
open Emgu.CV;
open Emgu.CV.UI;
open Emgu.CV.CvEnum;
open Emgu.CV.Structure;
open DojutsifyMe.FaceDetection;
open FSharp.Control.Reactive;
open FSharpx.Reader

let display (imageBox:ImageBox) (image:UMat) = 
    imageBox.Image <- image

let retrieveFrame channel (capture : VideoCapture) =
    let frame = new UMat()
    match capture.Retrieve(frame, channel) with
        | true -> Some frame
        | _ -> None 

let drawRectangle color (frame:UMat) rectangle =
    CvInvoke.Rectangle(frame, rectangle, Bgr(color).MCvScalar, 2)

[<EntryPoint>]
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")

    let mainBox = new ImageBox(Location=Point(0,0), Size=Size(500,500), Image=null)
    let secondBox = new ImageBox(Location=Point(500,0), Size=Size(300,250), Image=null)
    let thirdBox = new ImageBox(Location=Point(500,250), Size=Size(300,250), Image=null)
    
    form.Controls.AddRange([|mainBox;secondBox;thirdBox|])

    let processFrame frame =
        let grayScaled = frame |> grayScale
        let equalized = grayScaled |> equalizeHistogram
        let faces = equalized |> detectFace
        let eyes = faces |> List.map (detectEyes equalized)
        
        faces |> List.iter (drawRectangle Color.Red frame)
        eyes |> List.concat |> List.iter (drawRectangle Color.Blue frame)
        
        let (GrayScaled gray) = grayScaled in display secondBox gray
        let (EqualizedHistogram eq) = equalized in display thirdBox eq
        display mainBox frame

    let capture = new VideoCapture()
    capture.Start()
    
    use imageGrabbedObservable = 
        capture.ImageGrabbed |> 
            Observable.map (fun _ -> capture) |> 
            Observable.filter (fun capture -> capture.Ptr <> IntPtr.Zero) |> 
            Observable.map (retrieveFrame 0) |>
            Observable.subscribe (Option.map processFrame >> ignore)

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0