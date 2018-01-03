module DojutsifyMe.FaceDetection

open Emgu.CV;
open System.Drawing;
open DojutsifyMe.ImageProcessing

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

let tryExtractFace equalized =
    let faces = equalized |> detectFace
    let eyes = faces |> List.collect (detectEyes equalized)

    match faces, eyes with
        | ([head],[leftEye;rightEye]) -> Some (head,[leftEye;rightEye])
        | _ -> None