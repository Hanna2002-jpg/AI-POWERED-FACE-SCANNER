import React, { useState } from 'react';
import { Brain, Play, Pause, RotateCcw, Settings, TrendingUp, Clock, Zap } from 'lucide-react';

export function TrainingPipeline() {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(20);
  const [totalEpochs] = useState(20);

  const hyperparameters = {
    learningRate: 0.001,
    batchSize: 32,
    optimizer: 'Adam',
    scheduler: 'StepLR',
    augmentation: 'Enabled',
    freezeBackbone: false
  };

  const trainingLog = [
    { epoch: 20, loss: 0.28, accuracy: 0.942, lr: 0.0001, time: '14:32' },
    { epoch: 19, loss: 0.31, accuracy: 0.938, lr: 0.0001, time: '14:25' },
    { epoch: 18, loss: 0.33, accuracy: 0.935, lr: 0.0001, time: '14:18' },
    { epoch: 17, loss: 0.36, accuracy: 0.928, lr: 0.0001, time: '14:11' },
    { epoch: 16, loss: 0.38, accuracy: 0.924, lr: 0.0001, time: '14:04' }
  ];

  const modelArchitecture = {
    backbone: 'ResNet-50',
    embedding: 512,
    loss: 'ArcFace',
    margin: 0.5,
    scale: 64
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-4">Training Pipeline</h2>
        <p className="text-slate-300 max-w-2xl mx-auto">
          Advanced fine-tuning pipeline with transfer learning, custom loss functions, 
          and comprehensive monitoring for optimal face recognition performance.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Control Panel */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Brain className="h-5 w-5 mr-2 text-blue-400" />
            Training Control
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-slate-700/30 rounded-lg">
              <div>
                <p className="font-medium text-white">Training Status</p>
                <p className="text-sm text-slate-300">
                  {isTraining ? 'Training in progress...' : 'Training completed'}
                </p>
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={() => setIsTraining(!isTraining)}
                  className={`p-2 rounded-lg transition-colors ${
                    isTraining 
                      ? 'bg-red-600 hover:bg-red-700 text-white' 
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  {isTraining ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <button className="p-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors">
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-slate-300">Progress</span>
                <span className="text-blue-400">{currentEpoch}/{totalEpochs} epochs</span>
              </div>
              <div className="bg-slate-700 rounded-full h-2">
                <div 
                  className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }}
                />
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center p-3 bg-slate-700/30 rounded-lg">
                <div className="text-lg font-bold text-green-400">94.2%</div>
                <div className="text-xs text-slate-300">Current Accuracy</div>
              </div>
              <div className="text-center p-3 bg-slate-700/30 rounded-lg">
                <div className="text-lg font-bold text-blue-400">0.28</div>
                <div className="text-xs text-slate-300">Current Loss</div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Configuration */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Settings className="h-5 w-5 mr-2 text-purple-400" />
            Model Configuration
          </h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-purple-300 mb-2">Architecture</h4>
              <div className="space-y-2">
                {Object.entries(modelArchitecture).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-slate-300 capitalize">{key}:</span>
                    <span className="text-white font-medium">{value}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-purple-300 mb-2">Hyperparameters</h4>
              <div className="space-y-2">
                {Object.entries(hyperparameters).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-slate-300 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                    <span className="text-white font-medium">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Metrics */}
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 mr-2 text-green-400" />
          Training Metrics
        </h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-green-300 mb-3">Recent Training Log</h4>
            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="space-y-2">
                <div className="grid grid-cols-5 gap-2 text-xs font-medium text-slate-400 pb-2 border-b border-slate-600">
                  <span>Epoch</span>
                  <span>Loss</span>
                  <span>Accuracy</span>
                  <span>LR</span>
                  <span>Time</span>
                </div>
                {trainingLog.map((entry) => (
                  <div key={entry.epoch} className="grid grid-cols-5 gap-2 text-sm">
                    <span className="text-blue-400">{entry.epoch}</span>
                    <span className="text-red-400">{entry.loss}</span>
                    <span className="text-green-400">{entry.accuracy}</span>
                    <span className="text-purple-400">{entry.lr}</span>
                    <span className="text-slate-300">{entry.time}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-green-300 mb-3">Performance Indicators</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Zap className="h-4 w-4 text-yellow-400" />
                    <span className="text-slate-300">Training Speed</span>
                  </div>
                  <span className="text-yellow-400 font-medium">2.3 samples/sec</span>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-blue-400" />
                    <span className="text-slate-300">ETA</span>
                  </div>
                  <span className="text-blue-400 font-medium">Completed</span>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4 text-green-400" />
                    <span className="text-slate-300">Improvement</span>
                  </div>
                  <span className="text-green-400 font-medium">+2.1%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Strategy */}
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Fine-tuning Strategy</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg">
            <h4 className="font-semibold text-blue-300 mb-2">Transfer Learning</h4>
            <p className="text-sm text-slate-300">
              Pre-trained ResNet-50 backbone with custom ArcFace head for face recognition tasks.
            </p>
          </div>
          
          <div className="p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
            <h4 className="font-semibold text-green-300 mb-2">Gradual Unfreezing</h4>
            <p className="text-sm text-slate-300">
              Progressive unfreezing of layers to prevent overfitting and maintain learned features.
            </p>
          </div>
          
          <div className="p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
            <h4 className="font-semibold text-purple-300 mb-2">Data Augmentation</h4>
            <p className="text-sm text-slate-300">
              Random horizontal flips, rotations, and color jittering to improve generalization.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}