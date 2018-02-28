module DojutsifyMe.Utils

open System.Drawing

let mapTuple f (a,b) = (f a, f b)

let pointfToTuple (pointf:PointF) = (pointf.X, pointf.Y)

let tupleToPointf (x, y) = PointF(x, y)

let meanOfPoints (points:PointF list) = 
    points |> List.map pointfToTuple
        |> List.reduce (fun (accX, accY) (x, y) -> (accX + x, accY + y))
        |> mapTuple (fun point -> point / (points |> List.length |> float32))
        |> tupleToPointf