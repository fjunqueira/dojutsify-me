module DojutsifyMe.Utils

open System.Drawing
open Emgu.CV

let mapTuple f (a,b) = (f a, f b)

let pointfToTuple (pointf:PointF) = (pointf.X, pointf.Y)

let tupleToPointf (x, y) = PointF(x, y)

let meanOfPoints (points:PointF list) = 
    points |> List.map pointfToTuple
        |> List.reduce (fun (accX, accY) (x, y) -> (accX + x, accY + y))
        |> mapTuple (fun point -> point / (points |> List.length |> float32))
        |> tupleToPointf

let minMaxLoc image =

    let mutable minVal = ref 0.0
    let mutable maxVal = ref 0.0
    let mutable minLoc = ref <| Point()
    let mutable maxLoc = ref <| Point()
    
    CvInvoke.MinMaxLoc(image, minVal, maxVal, minLoc, maxLoc)        

    (minVal.Value, maxVal.Value, minLoc.Value, maxLoc.Value)