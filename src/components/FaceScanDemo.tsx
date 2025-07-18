import React, { useState, useRef, useCallback } from 'react';
import { 
  Upload, 
  Camera, 
  Scan, 
  CheckCircle, 
  AlertCircle, 
  Users, 
  Target,
  Eye,
  Download,
  Play,
  Pause
} from 'lucide-react';

interface ScanResult {
  id: string;
  confidence: number;
  identity: string;
  similarity: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  features: {
    age: number;
    gender: string;
    emotion: string;
  };
}

interface ComparisonResult {
  similarity: number;
  match: boolean;
  confidence: number;
  processingTime: number;
}

export function FaceScanDemo() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [comparisonImage, setComparisonImage] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [activeMode, setActiveMode] = useState<'single' | 'comparison' | 'realtime'>('single');
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const comparisonInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>, isComparison = false) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        if (isComparison) {
          setComparisonImage(result);
        } else {
          setSelectedImage(result);
        }
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const simulateFaceScan = useCallback(async () => {
    setIsScanning(true);
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate mock scan results
    const mockResults: ScanResult[] = [
      {
        id: '1',
        confidence: 0.94,
        identity: 'Person_001',
        similarity: 0.89,
        boundingBox: { x: 120, y: 80, width: 180, height: 220 },
        features: { age: 28, gender: 'Female', emotion: 'Neutral' }
      },
      {
        id: '2',
        confidence: 0.87,
        identity: 'Person_002',
        similarity: 0.76,
        boundingBox: { x: 350, y: 120, width: 160, height: 200 },
        features: { age: 35, gender: 'Male', emotion: 'Happy' }
      }
    ];
    
    setScanResults(mockResults);
    setIsScanning(false);
  }, []);

  const simulateComparison = useCallback(async () => {
    if (!selectedImage || !comparisonImage) return;
    
    setIsScanning(true);
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Generate mock comparison result
    const similarity = Math.random() * 0.4 + 0.6; // 0.6 - 1.0
    const mockResult: ComparisonResult = {
      similarity,
      match: similarity > 0.8,
      confidence: similarity * 0.95,
      processingTime: 1.2
    };
    
    setComparisonResult(mockResult);
    setIsScanning(false);
  }, [selectedImage, comparisonImage]);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsWebcamActive(true);
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsWebcamActive(false);
    }
  }, []);

  const renderSingleScanMode = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Image Upload */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Upload className="h-5 w-5 mr-2 text-blue-400" />
            Upload Image
          </h3>
          
          <div 
            className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors"
            onClick={() => fileInputRef.current?.click()}
          >
            {selectedImage ? (
              <img 
                src={selectedImage} 
                alt="Selected" 
                className="max-w-full max-h-64 mx-auto rounded-lg"
              />
            ) : (
              <div className="space-y-4">
                <Upload className="h-12 w-12 text-slate-400 mx-auto" />
                <p className="text-slate-300">Click to upload an image</p>
                <p className="text-sm text-slate-500">Supports JPG, PNG, WebP</p>
              </div>
            )}
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => handleImageUpload(e)}
            className="hidden"
          />
          
          <button
            onClick={simulateFaceScan}
            disabled={!selectedImage || isScanning}
            className="w-full mt-4 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center space-x-2"
          >
            {isScanning ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                <span>Scanning...</span>
              </>
            ) : (
              <>
                <Scan className="h-4 w-4" />
                <span>Start Face Scan</span>
              </>
            )}
          </button>
        </div>

        {/* Scan Results */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Target className="h-5 w-5 mr-2 text-green-400" />
            Scan Results
          </h3>
          
          {scanResults.length > 0 ? (
            <div className="space-y-4">
              {scanResults.map((result) => (
                <div key={result.id} className="bg-slate-700/30 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-white">{result.identity}</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      result.confidence > 0.9 ? 'bg-green-900/30 text-green-400' : 'bg-yellow-900/30 text-yellow-400'
                    }`}>
                      {(result.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-slate-400">Similarity:</span>
                      <span className="text-white ml-2">{(result.similarity * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-slate-400">Age:</span>
                      <span className="text-white ml-2">{result.features.age}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">Gender:</span>
                      <span className="text-white ml-2">{result.features.gender}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">Emotion:</span>
                      <span className="text-white ml-2">{result.features.emotion}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Eye className="h-12 w-12 text-slate-400 mx-auto mb-4" />
              <p className="text-slate-300">No scan results yet</p>
              <p className="text-sm text-slate-500">Upload an image and start scanning</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderComparisonMode = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* First Image */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-lg font-bold text-white mb-4">Reference Image</h3>
          
          <div 
            className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 transition-colors"
            onClick={() => fileInputRef.current?.click()}
          >
            {selectedImage ? (
              <img 
                src={selectedImage} 
                alt="Reference" 
                className="max-w-full max-h-48 mx-auto rounded-lg"
              />
            ) : (
              <div className="space-y-2">
                <Upload className="h-8 w-8 text-slate-400 mx-auto" />
                <p className="text-sm text-slate-300">Upload reference</p>
              </div>
            )}
          </div>
        </div>

        {/* Second Image */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-lg font-bold text-white mb-4">Comparison Image</h3>
          
          <div 
            className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 transition-colors"
            onClick={() => comparisonInputRef.current?.click()}
          >
            {comparisonImage ? (
              <img 
                src={comparisonImage} 
                alt="Comparison" 
                className="max-w-full max-h-48 mx-auto rounded-lg"
              />
            ) : (
              <div className="space-y-2">
                <Upload className="h-8 w-8 text-slate-400 mx-auto" />
                <p className="text-sm text-slate-300">Upload comparison</p>
              </div>
            )}
          </div>
        </div>

        {/* Comparison Result */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-lg font-bold text-white mb-4">Match Result</h3>
          
          {comparisonResult ? (
            <div className="space-y-4">
              <div className={`text-center p-4 rounded-lg ${
                comparisonResult.match ? 'bg-green-900/30 border border-green-500/30' : 'bg-red-900/30 border border-red-500/30'
              }`}>
                {comparisonResult.match ? (
                  <CheckCircle className="h-8 w-8 text-green-400 mx-auto mb-2" />
                ) : (
                  <AlertCircle className="h-8 w-8 text-red-400 mx-auto mb-2" />
                )}
                <p className={`font-semibold ${comparisonResult.match ? 'text-green-400' : 'text-red-400'}`}>
                  {comparisonResult.match ? 'MATCH' : 'NO MATCH'}
                </p>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Similarity:</span>
                  <span className="text-white">{(comparisonResult.similarity * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Confidence:</span>
                  <span className="text-white">{(comparisonResult.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Processing:</span>
                  <span className="text-white">{comparisonResult.processingTime}s</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <Target className="h-8 w-8 text-slate-400 mx-auto mb-2" />
              <p className="text-sm text-slate-300">Upload both images</p>
            </div>
          )}
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleImageUpload(e)}
        className="hidden"
      />
      
      <input
        ref={comparisonInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleImageUpload(e, true)}
        className="hidden"
      />

      <div className="text-center">
        <button
          onClick={simulateComparison}
          disabled={!selectedImage || !comparisonImage || isScanning}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-6 py-3 rounded-lg transition-colors flex items-center space-x-2 mx-auto"
        >
          {isScanning ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
              <span>Comparing...</span>
            </>
          ) : (
            <>
              <Scan className="h-4 w-4" />
              <span>Compare Faces</span>
            </>
          )}
        </button>
      </div>
    </div>
  );

  const renderRealtimeMode = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Camera className="h-5 w-5 mr-2 text-purple-400" />
          Real-time Face Scanning
        </h3>
        
        <div className="relative">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full max-w-2xl mx-auto rounded-lg bg-slate-900"
            style={{ aspectRatio: '16/9' }}
          />
          
          {!isWebcamActive && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 rounded-lg">
              <div className="text-center">
                <Camera className="h-16 w-16 text-slate-400 mx-auto mb-4" />
                <p className="text-slate-300 mb-4">Webcam not active</p>
                <button
                  onClick={startWebcam}
                  className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 mx-auto"
                >
                  <Play className="h-4 w-4" />
                  <span>Start Webcam</span>
                </button>
              </div>
            </div>
          )}
        </div>
        
        {isWebcamActive && (
          <div className="flex justify-center space-x-4 mt-4">
            <button
              onClick={stopWebcam}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
            >
              <Pause className="h-4 w-4" />
              <span>Stop Webcam</span>
            </button>
            
            <button
              onClick={simulateFaceScan}
              disabled={isScanning}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
            >
              <Scan className="h-4 w-4" />
              <span>Scan Current Frame</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white mb-4">
          Face Scan Master - Live Demo
        </h1>
        <p className="text-slate-300 text-lg max-w-2xl mx-auto">
          Experience advanced face scanning technology with real-time detection, 
          comparison, and analysis capabilities powered by state-of-the-art AI models.
        </p>
      </div>

      {/* Mode Selection */}
      <div className="flex justify-center">
        <div className="flex space-x-1 bg-slate-800/50 rounded-lg p-1">
          <button
            onClick={() => setActiveMode('single')}
            className={`px-4 py-2 rounded-lg transition-all duration-200 ${
              activeMode === 'single'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-slate-700/50'
            }`}
          >
            Single Scan
          </button>
          <button
            onClick={() => setActiveMode('comparison')}
            className={`px-4 py-2 rounded-lg transition-all duration-200 ${
              activeMode === 'comparison'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-slate-700/50'
            }`}
          >
            Face Comparison
          </button>
          <button
            onClick={() => setActiveMode('realtime')}
            className={`px-4 py-2 rounded-lg transition-all duration-200 ${
              activeMode === 'realtime'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-slate-700/50'
            }`}
          >
            Real-time Scan
          </button>
        </div>
      </div>

      {/* Mode Content */}
      {activeMode === 'single' && renderSingleScanMode()}
      {activeMode === 'comparison' && renderComparisonMode()}
      {activeMode === 'realtime' && renderRealtimeMode()}

      {/* Performance Stats */}
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Performance Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <div className="text-2xl font-bold text-blue-400">94.2%</div>
            <div className="text-sm text-slate-300">Accuracy</div>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <div className="text-2xl font-bold text-green-400">1.2s</div>
            <div className="text-sm text-slate-300">Avg Processing</div>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <div className="text-2xl font-bold text-purple-400">512</div>
            <div className="text-sm text-slate-300">Embedding Size</div>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <div className="text-2xl font-bold text-orange-400">30fps</div>
            <div className="text-sm text-slate-300">Real-time Rate</div>
          </div>
        </div>
      </div>
    </div>
  );
}