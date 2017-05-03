module DojutsifyMe.Main

open System;
open System.Drawing;
open System.Windows.Forms;
open Emgu.CV;
open Emgu.CV.UI;
open Emgu.CV.CvEnum;
open Emgu.CV.Structure;
open DojutsifyMe.FaceDetection;

let retrieveFrame channel (capture : VideoCapture) =
    let frame = new UMat()
    let success = capture.Retrieve(frame, channel)
    (success, frame)

let drawRectangle frame rectangle color =
    CvInvoke.Rectangle(frame, rectangle, Bgr(color).MCvScalar, 2)

let drawFaceRectangle frame (face, eyes) =
    drawRectangle frame face Color.Red
    List.iter (fun eye -> drawRectangle frame eye Color.Blue) eyes

[<EntryPoint>]
[<STAThread>]
let main args = 
    let form = new Form(Width=800, Height=500, Name="Dojutsify Me")
    let imageBox = new ImageBox(Location=Point(0,0), Size=Size(800,500), Image=null)
    form.Controls.Add(imageBox)

    let capture = new VideoCapture()
    capture.Start()
    
    use imageGrabbedEvent = 
        capture.ImageGrabbed |> 
            Observable.map (fun _ -> capture) |> 
            Observable.filter (fun capture -> capture.Ptr <> IntPtr.Zero) |> 
            Observable.map (retrieveFrame 0) |>
            Observable.map (fun retrieved -> match retrieved with
                                                | (true, frame) -> frame, detectFace frame
                                                | (_, frame) -> frame , []) |>
            Observable.subscribe (fun (frame, faces) -> List.iter (drawFaceRectangle frame) faces
                                                        imageBox.Image <- frame)

    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    Application.Run(form)

    0