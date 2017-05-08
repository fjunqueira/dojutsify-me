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
open FSharpx.Choice
open FSharpx

let display (imageBox:ImageBox) (image:Mat) = 
    imageBox.Image <- image

let retrieveFrame channel (capture:VideoCapture) =
    let frame = new Mat()
    (capture.Retrieve(frame, channel),frame)

let drawRectangle color (frame:Mat) rectangle =
    CvInvoke.Rectangle(frame, rectangle, Bgr(color).MCvScalar, 2)

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
    
    let imageGrabbed = capture.ImageGrabbed

    imageGrabbed |>
        Observable.map (fun _ -> capture) |> 
        Observable.filter (fun capture -> capture.Ptr <> IntPtr.Zero) |> 
        Observable.map (retrieveFrame 0) |>
        Observable.filter fst |> 
        Observable.map (snd >> extractFace) |>
        //Observable.flatmap (fun face -> imageGrabbed |> Observable.map (fun _ -> face)) |>
        Observable.subscribe (fun face -> match face with
                                            | Choice1Of2 ([head],eye1::eyeN) -> printfn "%s" "Success"
                                            | Choice2Of2 message -> printfn "%s" message
                                            | _ -> ()) |> ignore

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0