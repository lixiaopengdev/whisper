import Foundation
import AVFAudio
import onnxruntime_objc


public class SpeakerEmbedding {
    private var _modelHandler: SpeakerEmbeddingModelHandler?
    private let expectedFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)
    
    public init() {
        loadModel()
    }
    
    
    
    private func loadModel() {
        guard _modelHandler == nil else {
            return
        }
        
        _modelHandler = SpeakerEmbeddingModelHandler(modelFilename: "speaker_embedding0", modelExtension: "onnx", threadCount: 4)
    }
    
    private func _checkAudioFormat(pcmFormat: AVAudioFormat) -> Bool {
        // 检查采样率是否匹配
        guard pcmFormat.sampleRate == expectedFormat!.sampleRate else {
            return false
        }
        
        // 检查通道数是否匹配
        guard pcmFormat.channelCount == expectedFormat!.channelCount else {
            return false
        }
        
        // 检查位深度是否匹配
        guard pcmFormat.commonFormat == expectedFormat!.commonFormat else {
            return false
        }
        
        return true
    }
}


public extension SpeakerEmbedding {
    func extractFeature(buffer: AVAudioPCMBuffer, range: Range<Int> ) -> [Float]? {
        guard let modelHandler = _modelHandler else {
            return nil
        }
        guard _checkAudioFormat(pcmFormat: buffer.format) else {
            return nil
        }
        let channelData: UnsafePointer<UnsafeMutablePointer<Float32>> = buffer.floatChannelData!
        let channelPointer: UnsafeMutablePointer<Float32> = channelData[0]
        let frameLength = Int(buffer.frameLength)
        
        let startIndex = range.lowerBound
        let count = min(range.count, (frameLength - startIndex)) * MemoryLayout<Float32>.stride
        
        let pointer: UnsafeMutablePointer<Float32> = channelPointer.advanced(by: startIndex)
        let data = Data(bytes: pointer, count: count)
        
        return extractFeature(data: data)
    }
    
    func extractFeature(data: Data) -> [Float]? {
        guard let modelHandler = _modelHandler else {
            return nil
        }
        
        return modelHandler.prediction(x: data)
    }
    
}

