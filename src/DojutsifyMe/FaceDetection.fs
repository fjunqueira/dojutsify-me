module DojutsifyMe.FaceDetection

open Emgu.CV;
open Emgu.CV.Structure;
open System.Drawing;
open FSharpx.Reader

let grayScale frame = 
    let grayFrame = new UMat()
    CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray)
    grayFrame

let equalizeHistogram (grayScale:UMat) = 
    CvInvoke.EqualizeHist(grayScale, grayScale)
    grayScale

let detectEyes grayFrame (face:Rectangle) =
    use faceRegion = new UMat(grayFrame, face)
    use eyeCascade = new CascadeClassifier("haarcascade_eye.xml")
    
    eyeCascade.DetectMultiScale(faceRegion, 1.1, 10, Size(20, 20)) |>
        Array.map (fun (eye:Rectangle) -> eye.Offset(Point(face.X, face.Y)); eye) |> 
        Array.toList

let detectFace (grayScaledImage:UMat) = 
    use faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml")
    
    faceCascade.DetectMultiScale(grayScaledImage, 1.1, 10, Size(20, 20)) |> 
        Array.toList