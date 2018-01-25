module DojutsifyMe.FaceDetection

open Emgu.CV;
open System.Drawing;
open DojutsifyMe.ImageProcessing

let detectEyes (EqualizedHistogram frame) (face:Rectangle) =
    use faceRegion = new Mat(frame, face)
    use eyeCascade = new CascadeClassifier("haarcascade_eye.xml")
    
    eyeCascade.DetectMultiScale(faceRegion, 1.1, 10, Size(10, 10), Size(70, 70)) |>
        Array.map (fun (eye:Rectangle) -> eye.Offset(Point(face.X, face.Y)); eye) |> 
        Array.toList

let detectFace (EqualizedHistogram frame) = 
    use faceCascade = new CascadeClassifier("haarcascade_frontalface_alt.xml")
    
    faceCascade.DetectMultiScale(frame, 1.1, 2, Size(150, 150)) |> 
        Array.toList

let tryFindingEyes equalized face = 
    let eyes = detectEyes equalized face

    match eyes with
    | [_; _] -> Some eyes
    | _ -> None

let tryFindingFace equalized =
    let faces = equalized |> detectFace

    match faces with
    | [head] -> Some head
    | _ -> None