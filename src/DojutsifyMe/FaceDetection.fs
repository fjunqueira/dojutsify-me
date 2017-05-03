module DojutsifyMe.FaceDetection

open Emgu.CV;
open Emgu.CV.Structure;
open System.Drawing;

let detectEyes face (grayFrame:UMat) =
    use faceRegion = new UMat(grayFrame, face)
    use eyeCascade = new CascadeClassifier("haarcascade_eye.xml")
    
    eyeCascade.DetectMultiScale(faceRegion, 1.1, 10, Size(20, 20)) |>
        Array.map (fun (eye:Rectangle) -> eye.Offset(Point(face.X, face.Y)); eye) |> 
        Array.toList

let detectFace (frame:IInputArray) = 
    use grayFrame = new UMat()
    use faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml")

    CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray)
    CvInvoke.EqualizeHist(grayFrame, grayFrame)

    faceCascade.DetectMultiScale(grayFrame, 1.1, 10, Size(20, 20)) |> 
        Array.map (fun face -> (face, detectEyes face grayFrame)) |> 
        Array.toList