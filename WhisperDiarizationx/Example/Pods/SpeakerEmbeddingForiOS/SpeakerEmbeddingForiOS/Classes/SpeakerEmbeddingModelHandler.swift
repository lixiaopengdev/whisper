//
//  ModelHandler.swift
//  SpeakerEmbeddingForiOS
//
//  Created by fuhao on 2023/4/27.
//

import Foundation
import Accelerate
import AVFoundation
import CoreImage
import Darwin
import Foundation
import UIKit
import onnxruntime_objc



// EmResult struct
struct EmResult {
    let processTimeMs: Double
    let feature: [Float]
}



class SpeakerEmbeddingModelHandler: NSObject {
    // MARK: - Inference Properties
    let threadCount: Int32
    let threadCountLimit = 10
    
    // MARK: - Model Parameters
    let batchSize = 1
    
    

    
    
    
    private var session: ORTSession
    private var env: ORTEnv
    
    init?(modelFilename: String, modelExtension: String, threadCount: Int32 = 1) {
        guard let associateBundleURL2 = Bundle.main.url(forResource: "SpeakerEmbedding", withExtension: "bundle") else {
            return nil
        }
        
        guard let podBundle = Bundle(url: associateBundleURL2) else {
            return nil
        }
        
        guard let modelPath = podBundle.path(forResource: modelFilename, ofType: modelExtension) else {
            print("Failed to get model file path with name: \(modelFilename).")
            return nil
        }
        
        
        
        self.threadCount = threadCount
        do {
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.warning)
            try options.setIntraOpNumThreads(threadCount)
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch {
            print("Failed to create ORTSession.")
            return nil
        }
        
        super.init()
    }
    
    
    
    func _parseToFloatArray(value: ORTValue?) throws -> [Float] {
        guard let rawOutputValue = value else {
            throw OrtModelError.error("failed to get model output")
        }
//        let outputTensor = try rawOutputValue.tensorData()
//        print(outputTensor.description)
        
        
        let rawOutputData = try rawOutputValue.tensorData() as Data
//
//        let hexString = rawOutputData.prefix(8).map { String(format: "%02hhx ", $0) }.joined()
//        print(hexString)
        
//        let itemCount = rawOutputData.count / MemoryLayout<Float>.size
////        let rawOutputData = Data(bytes: [0x3F,0x00,0x00,0x00,0x00,0x00,0x00,0x3F])
//
//
//        var floatArray = [Float](repeating: 0, count: itemCount) // 创建一个初始值为0的数组来存储浮点数
//        rawOutputData.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
//            let floatPointer:UnsafeBufferPointer<UInt32> = bytes.bindMemory(to: UInt32.self)
//
//            for i in 0..<itemCount {
//                let iii = floatPointer.baseAddress?.advanced(by: i).pointee
//                let v = Float(bitPattern: iii!)
//                floatArray[i] = v
//            }
//        }

        
        let count = rawOutputData.count / MemoryLayout<Float>.size
        let floatArray = rawOutputData.withUnsafeBytes { (pointer: UnsafePointer<Float32>) -> [Float32] in
            let buffer = UnsafeBufferPointer(start: pointer,count: count)
            return Array<Float32>(buffer)
        }

        return floatArray
    }
    
    func _prediction(inputTensors: [ORTValue]) throws -> EmResult {
        let inputNames = ["wav"]
        let outputNames: Set<String> = ["output"]
        
        guard inputTensors.count == inputNames.count else {
            throw OrtModelError.error("inputTensors.count != inputNames.count")
        }
        
        let inputDic = Dictionary(uniqueKeysWithValues: zip(inputNames, inputTensors))
        
        let interval: TimeInterval
        let startDate = Date()
        let outputs:[String: ORTValue] = try session.run(withInputs: inputDic,
                                      outputNames: outputNames,
                                      runOptions: nil)
        interval = Date().timeIntervalSince(startDate) * 1000
        
        
        let feature = try _parseToFloatArray(value: outputs["output"])
        
        // Return ORT SessionRun result
        return EmResult(processTimeMs: interval, feature: feature)
    }
    
    
    

    
}




extension SpeakerEmbeddingModelHandler {
    func prediction(x: Data) -> [Float]?{
        do {
            let size = x.count / MemoryLayout<Float>.size
            let inputShape: [NSNumber] = [batchSize as NSNumber,
                                          size as NSNumber]
            let xTensor:ORTValue = try ORTValue(tensorData: NSMutableData(data: x),
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape)
//            let inputTensor = try xTensor.tensorData()
//            print(inputTensor.description)
           
            let inputTensors:[ORTValue] = [xTensor]
            let predictionResult = try _prediction(inputTensors: inputTensors)
            return predictionResult.feature
        } catch {
            print("Unknown error: \(error)")
        }
        
        return nil
    }
    
    
}
