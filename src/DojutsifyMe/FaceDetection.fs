module DojutsifyMe.FaceDetection

open Emgu.CV;
open Emgu.CV.Structure;
open System.Drawing;
open FSharpx.Reader

type GrayScaled = GrayScaled of UMat
type EqualizedHistogram = EqualizedHistogram of UMat

let grayScale frame = 
    let grayFrame = new UMat()
    CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray)
    GrayScaled grayFrame

let equalizeHistogram grayScale = 
    let (GrayScaled frame) = grayScale
    let equalizedFrame = new UMat()
    frame.CopyTo(equalizedFrame)
    CvInvoke.EqualizeHist(equalizedFrame, equalizedFrame)
    EqualizedHistogram equalizedFrame

let detectEyes equalized (face:Rectangle) =
    let (EqualizedHistogram frame) = equalized
    use faceRegion = new UMat(frame, face)
    use eyeCascade = new CascadeClassifier("haarcascade_eye.xml")
    
    eyeCascade.DetectMultiScale(faceRegion, 1.1, 10, Size(20, 20)) |>
        Array.map (fun (eye:Rectangle) -> eye.Offset(Point(face.X, face.Y)); eye) |> 
        Array.toList

let detectFace equalized = 
    let (EqualizedHistogram frame) = equalized
    use faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml")
    
    faceCascade.DetectMultiScale(frame, 1.1, 10, Size(20, 20)) |> 
        Array.toList