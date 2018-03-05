module DojutsifyMe.FaceDetection

open Emgu.CV;
open System.Drawing;
open DojutsifyMe.ImageProcessing
open DojutsifyMe.Utils

let faceClassifier = new CascadeClassifier("haarcascade_frontalface_alt.xml")

let leftEyeClassifier = new CascadeClassifier("haarcascade_lefteye_2splits.xml")

let rightEyeClassifier = new CascadeClassifier("haarcascade_righteye_2splits.xml")

let detectEyes (GrayScaled frame) eyeArea (classifier:CascadeClassifier) =
    use eyeROI = new Mat(frame, eyeArea)
     
    classifier.DetectMultiScale(eyeROI, 1.15, 2, Size(30, 30)) |>
        Array.map (fun eye -> eye.Offset(Point(eyeArea.X, eyeArea.Y)); eye) |>
        Array.toList

let detectFaces (GrayScaled frame) = 
    faceClassifier.DetectMultiScale(frame, 1.1, 2, Size(150, 150)) |> 
        Array.toList

let calculateEyeArea (face:Rectangle) =
    let leftEyeArea =  Rectangle(face.X + face.Width / 16 + (face.Width - 2 * face.Width / 16) / 2, 
                                    (int) ((float)face.Y + ((float) face.Height / 4.5)), 
                                    (face.Width - 2 * face.Width / 16) / 2, 
                                    (face.Height / 3))

    let rightEyeArea = Rectangle(face.X + face.Width / 16,
                                    (int) ((float)face.Y + ((float) face.Height / 4.5)), 
                                    (face.Width - 2 * face.Width / 16) / 2, 
                                    (face.Height / 3))

    (leftEyeArea, rightEyeArea)

let refineEyeArea (GrayScaled frame) size (eyeArea:Rectangle) =
    
    let refinedArea = Rectangle(eyeArea.X, (int) ((float) eyeArea.Y + (float) eyeArea.Height * 0.4), eyeArea.Width, (int) ((float)eyeArea.Height * 0.6));
    use refinedAreaROI = new Mat(frame, refinedArea)
    let (_, _, minLoc, _) = minMaxLoc refinedAreaROI
    let iris = Point(minLoc.X + refinedArea.X, minLoc.Y + refinedArea.Y)
    // printfn "Refined area height %d" refinedArea.Height
    // printfn "Refined area width %d" refinedArea.Width

    //ATTEMPT TO MOVE AROUND REDUCING STD DEVIATION
    Rectangle((int) iris.X - size / 2, (int) iris.Y - size / 2, size, size)
    //refinedArea

let tryFindingEyes frame faceArea = 
    
    let (leftEyeArea, rightEyeArea) = calculateEyeArea faceArea

    let detectEyeInFrame = detectEyes frame

    let leftEye =  detectEyeInFrame leftEyeArea  leftEyeClassifier
    let rightEye = detectEyeInFrame rightEyeArea rightEyeClassifier

    let eyes = match (leftEye, rightEye) with
               | ([left], [right]) -> Some (left, right)
               | _ -> None

    Option.map (mapTuple <| refineEyeArea frame 10) eyes
    //eyes

let tryFindingFace frame =
    frame |> detectFaces 
        |> List.sortByDescending (fun face -> face.Width * face.Height) 
        |> List.tryHead