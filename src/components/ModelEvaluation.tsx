import React, { useState } from 'react';
import { Target, BarChart3, TrendingUp, Award, AlertCircle, CheckCircle } from 'lucide-react';

export function ModelEvaluation() {
  const [activeTab, setActiveTab] = useState('metrics');

  const evaluationResults = {
    accuracy: 0.942,
    precision: 0.938,
    recall: 0.945,
    f1Score: 0.941,
    auc: 0.984,
    eer: 0.058
  };

  const modelComparison = [
    { name: 'ArcFace (Fine-tuned)', accuracy: 0.942, precision: 0.938, recall: 0.945, status: 'best' },
    { name: 'ArcFace (Baseline)', accuracy: 0.921, precision: 0.918, recall: 0.923, status: 'baseline' },
    { name: 'InsightFace', accuracy: 0.915, precision: 0.912, recall: 0.917, status: 'comparison' },
    { name: 'FaceNet', accuracy: 0.898, precision: 0.895, recall: 0.901, status: 'comparison' }
  ];

  const confusionMatrix = {
    truePositives: 1387,
    falsePositives: 89,
    falseNegatives: 85,
    trueNegatives: 1439
  };

  const thresholdAnalysis = [
    { threshold: 0.3, accuracy: 0.891, precision: 0.923, recall: 0.856 },
    { threshold: 0.4, accuracy: 0.918, precision: 0.935, recall: 0.898 },
    { threshold: 0.5, accuracy: 0.942, precision: 0.938, recall: 0.945 },
    { threshold: 0.6, accuracy: 0.935, precision: 0.952, recall: 0.916 },
    { threshold: 0.7, accuracy: 0.912, precision: 0.968, recall: 0.858 }
  ];

  const renderMetrics = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-blue-900/20 border border-blue-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Accuracy</p>
              <p className="text-2xl font-bold text-blue-400">{(evaluationResults.accuracy * 100).toFixed(1)}%</p>
            </div>
            <Target className="h-8 w-8 text-blue-400" />
          </div>
        </div>
        
        <div className="bg-green-900/20 border border-green-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Precision</p>
              <p className="text-2xl font-bold text-green-400">{(evaluationResults.precision * 100).toFixed(1)}%</p>
            </div>
            <Award className="h-8 w-8 text-green-400" />
          </div>
        </div>
        
        <div className="bg-purple-900/20 border border-purple-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Recall</p>
              <p className="text-2xl font-bold text-purple-400">{(evaluationResults.recall * 100).toFixed(1)}%</p>
            </div>
            <TrendingUp className="h-8 w-8 text-purple-400" />
          </div>
        </div>
        
        <div className="bg-orange-900/20 border border-orange-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">F1-Score</p>
              <p className="text-2xl font-bold text-orange-400">{(evaluationResults.f1Score * 100).toFixed(1)}%</p>
            </div>
            <BarChart3 className="h-8 w-8 text-orange-400" />
          </div>
        </div>
        
        <div className="bg-cyan-900/20 border border-cyan-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">AUC</p>
              <p className="text-2xl font-bold text-cyan-400">{(evaluationResults.auc * 100).toFixed(1)}%</p>
            </div>
            <TrendingUp className="h-8 w-8 text-cyan-400" />
          </div>
        </div>
        
        <div className="bg-red-900/20 border border-red-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">EER</p>
              <p className="text-2xl font-bold text-red-400">{(evaluationResults.eer * 100).toFixed(1)}%</p>
            </div>
            <AlertCircle className="h-8 w-8 text-red-400" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4">Confusion Matrix</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-900/20 border border-green-500/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-green-400">{confusionMatrix.truePositives}</div>
              <div className="text-sm text-slate-300">True Positives</div>
            </div>
            <div className="bg-red-900/20 border border-red-500/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-red-400">{confusionMatrix.falsePositives}</div>
              <div className="text-sm text-slate-300">False Positives</div>
            </div>
            <div className="bg-yellow-900/20 border border-yellow-500/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-yellow-400">{confusionMatrix.falseNegatives}</div>
              <div className="text-sm text-slate-300">False Negatives</div>
            </div>
            <div className="bg-blue-900/20 border border-blue-500/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-400">{confusionMatrix.trueNegatives}</div>
              <div className="text-sm text-slate-300">True Negatives</div>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4">ROC Curve Analysis</h3>
          <div className="h-48 bg-slate-900/50 rounded-lg flex items-center justify-center">
            <svg className="w-full h-full" viewBox="0 0 200 150">
              <path 
                d="M 20 130 Q 30 120 50 100 Q 80 70 120 50 Q 150 30 180 20"
                fill="none" 
                stroke="#22c55e" 
                strokeWidth="2"
              />
              <path 
                d="M 20 130 L 180 20"
                fill="none" 
                stroke="#64748b" 
                strokeWidth="1" 
                strokeDasharray="5,5"
              />
              <text x="100" y="145" textAnchor="middle" className="text-xs fill-slate-400">
                False Positive Rate
              </text>
              <text x="10" y="75" textAnchor="middle" className="text-xs fill-slate-400" transform="rotate(-90 10 75)">
                True Positive Rate
              </text>
            </svg>
          </div>
          <div className="mt-4 text-center">
            <span className="text-green-400 font-semibold">AUC = {evaluationResults.auc.toFixed(3)}</span>
            <span className="text-slate-300 text-sm ml-2">(Excellent Performance)</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderComparison = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Model Performance Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-600">
                <th className="text-left py-3 px-4 text-slate-300">Model</th>
                <th className="text-center py-3 px-4 text-slate-300">Accuracy</th>
                <th className="text-center py-3 px-4 text-slate-300">Precision</th>
                <th className="text-center py-3 px-4 text-slate-300">Recall</th>
                <th className="text-center py-3 px-4 text-slate-300">Status</th>
              </tr>
            </thead>
            <tbody>
              {modelComparison.map((model) => (
                <tr key={model.name} className="border-b border-slate-700/50">
                  <td className="py-3 px-4 text-white font-medium">{model.name}</td>
                  <td className="py-3 px-4 text-center">
                    <span className={`font-semibold ${model.status === 'best' ? 'text-green-400' : 'text-slate-300'}`}>
                      {(model.accuracy * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`font-semibold ${model.status === 'best' ? 'text-green-400' : 'text-slate-300'}`}>
                      {(model.precision * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`font-semibold ${model.status === 'best' ? 'text-green-400' : 'text-slate-300'}`}>
                      {(model.recall * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      model.status === 'best' 
                        ? 'bg-green-900/30 text-green-400 border border-green-500/30'
                        : model.status === 'baseline'
                        ? 'bg-blue-900/30 text-blue-400 border border-blue-500/30'
                        : 'bg-slate-700/30 text-slate-400 border border-slate-600/30'
                    }`}>
                      {model.status === 'best' ? 'Best' : model.status === 'baseline' ? 'Baseline' : 'Comparison'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Performance Improvements</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">Accuracy Gain</p>
                <p className="text-2xl font-bold text-green-400">+2.1%</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-400" />
            </div>
          </div>
          
          <div className="p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">Precision Gain</p>
                <p className="text-2xl font-bold text-blue-400">+2.0%</p>
              </div>
              <CheckCircle className="h-8 w-8 text-blue-400" />
            </div>
          </div>
          
          <div className="p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">Recall Gain</p>
                <p className="text-2xl font-bold text-purple-400">+2.2%</p>
              </div>
              <CheckCircle className="h-8 w-8 text-purple-400" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderThresholds = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Threshold Analysis</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-600">
                <th className="text-left py-3 px-4 text-slate-300">Threshold</th>
                <th className="text-center py-3 px-4 text-slate-300">Accuracy</th>
                <th className="text-center py-3 px-4 text-slate-300">Precision</th>
                <th className="text-center py-3 px-4 text-slate-300">Recall</th>
              </tr>
            </thead>
            <tbody>
              {thresholdAnalysis.map((threshold) => (
                <tr key={threshold.threshold} className={`border-b border-slate-700/50 ${
                  threshold.threshold === 0.5 ? 'bg-green-900/10' : ''
                }`}>
                  <td className="py-3 px-4 text-white font-medium">{threshold.threshold.toFixed(1)}</td>
                  <td className="py-3 px-4 text-center">
                    <span className={`font-semibold ${threshold.threshold === 0.5 ? 'text-green-400' : 'text-slate-300'}`}>
                      {(threshold.accuracy * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`font-semibold ${threshold.threshold === 0.5 ? 'text-green-400' : 'text-slate-300'}`}>
                      {(threshold.precision * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`font-semibold ${threshold.threshold === 0.5 ? 'text-green-400' : 'text-slate-300'}`}>
                      {(threshold.recall * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
          <p className="text-green-300 text-sm">
            <strong>Optimal Threshold:</strong> 0.5 provides the best balance between precision and recall, 
            achieving maximum accuracy of 94.2%.
          </p>
        </div>
      </div>
    </div>
  );

  const tabs = [
    { id: 'metrics', label: 'Performance Metrics', icon: Target },
    { id: 'comparison', label: 'Model Comparison', icon: BarChart3 },
    { id: 'thresholds', label: 'Threshold Analysis', icon: TrendingUp }
  ];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-4">Model Evaluation</h2>
        <p className="text-slate-300 max-w-2xl mx-auto">
          Comprehensive evaluation results with detailed metrics, model comparisons, 
          and threshold analysis for optimal face recognition performance.
        </p>
      </div>

      <div className="flex justify-center">
        <div className="flex space-x-1 bg-slate-800/50 rounded-lg p-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:bg-slate-700/50'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {activeTab === 'metrics' && renderMetrics()}
      {activeTab === 'comparison' && renderComparison()}
      {activeTab === 'thresholds' && renderThresholds()}
    </div>
  );
}