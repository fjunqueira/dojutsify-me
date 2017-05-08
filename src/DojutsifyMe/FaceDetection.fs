module DojutsifyMe.FaceDetection

open Emgu.CV;
open Emgu.CV.Structure;
open System.Drawing;

type GrayScaled = GrayScaled of Mat
type EqualizedHistogram = EqualizedHistogram of Mat

let grayScale frame = 
    let grayFrame = new Mat()
    CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray)
    GrayScaled grayFrame

let equalizeHistogram (GrayScaled frame) = 
    let equalizedFrame = new Mat()
    frame.CopyTo(equalizedFrame)
    CvInvoke.EqualizeHist(equalizedFrame, equalizedFrame)
    EqualizedHistogram equalizedFrame

let detectEyes (EqualizedHistogram frame) (face:Rectangle) =
    use faceRegion = new Mat(frame, face)
    use eyeCascade = new CascadeClassifier("haarcascade_eye.xml")
    
    eyeCascade.DetectMultiScale(faceRegion, 1.1, 10, Size(20, 20)) |>
        Array.map (fun (eye:Rectangle) -> eye.Offset(Point(face.X, face.Y)); eye) |> 
        Array.toList

let detectFace (EqualizedHistogram frame) = 
    use faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml")
    
    faceCascade.DetectMultiScale(frame, 1.1, 10, Size(20, 20)) |> 
        Array.toList

let Success = Choice1Of2
let Failure = Choice2Of2

let extractFace frame =

    let equalized = frame |> grayScale |> equalizeHistogram
    let faces = equalized |> detectFace
    let eyes = faces |> List.map (detectEyes equalized) |> List.concat

    match faces, eyes with
        | ([head],[leftEye;rightEye]) as data -> Success data
        | ([head],[eye]) as data -> Success data
        | ([], _) -> Failure "No head found"
        | (_, []) -> Failure "No eyes found"
        | (head1::head2::headN, _) -> Failure "Too many heads found"
        | (_, eye1::eye2::eye3::eyeN) -> Failure "Too many eyes found"