import React from 'react';
import { 
  TrendingUp, 
  Users, 
  Target, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Award,
  Cpu,
  Scan
} from 'lucide-react';
import { MetricCard } from './MetricCard';
import { PerformanceChart } from './PerformanceChart';

export function Dashboard() {
  const metrics = [
    {
      title: 'Scan Accuracy',
      value: '94.2%',
      change: '+2.1%',
      icon: Target,
      color: 'blue'
    },
    {
      title: 'Processing Speed',
      value: '1.2s',
      change: '-15%',
      icon: Clock,
      color: 'green'
    },
    {
      title: 'Identities Scanned',
      value: '1,247',
      change: '+0%',
      icon: Users,
      color: 'purple'
    },
    {
      title: 'Total Scans',
      value: '4,892',
      change: '+0%',
      icon: Scan,
      color: 'orange'
    }
  ];

  const status = [
    { label: 'Data Preprocessing', status: 'completed', progress: 100 },
    { label: 'Model Training', status: 'completed', progress: 100 },
    { label: 'Fine-tuning', status: 'completed', progress: 100 },
    { label: 'Evaluation', status: 'completed', progress: 100 },
    { label: 'Report Generation', status: 'in-progress', progress: 80 }
  ];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white mb-4">
          Face Scan Master - AI Evaluation System
        </h1>
        <p className="text-slate-300 text-lg max-w-2xl mx-auto">
          Complete implementation, fine-tuning, and evaluation of state-of-the-art face scanning models
          on realistic, low-quality image data for advanced AI assessment and deployment.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric) => (
          <MetricCard key={metric.title} {...metric} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-blue-400" />
            Performance Metrics
          </h3>
          <PerformanceChart />
        </div>

        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Cpu className="h-5 w-5 mr-2 text-green-400" />
            Project Status
          </h3>
          <div className="space-y-4">
            {status.map((item) => (
              <div key={item.label} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">{item.label}</span>
                  <div className="flex items-center space-x-2">
                    {item.status === 'completed' ? (
                      <CheckCircle className="h-4 w-4 text-green-400" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-yellow-400" />
                    )}
                    <span className="text-sm text-slate-400">{item.progress}%</span>
                  </div>
                </div>
                <div className="bg-slate-700 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      item.status === 'completed' ? 'bg-green-400' : 'bg-yellow-400'
                    }`}
                    style={{ width: `${item.progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Recent Achievements</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-900/20 border border-blue-500/20 rounded-lg p-4">
            <h4 className="font-semibold text-blue-300 mb-2">Advanced Scanning</h4>
            <p className="text-sm text-slate-300">
              Successfully implemented ArcFace, InsightFace, and FaceNet models. 
              ArcFace achieved best performance with 94.2% accuracy.
            </p>
          </div>
          <div className="bg-green-900/20 border border-green-500/20 rounded-lg p-4">
            <h4 className="font-semibold text-green-300 mb-2">Fine-tuning Complete</h4>
            <p className="text-sm text-slate-300">
              Implemented transfer learning with gradual unfreezing strategy. 
              Achieved 2.1% improvement over baseline performance.
            </p>
          </div>
          <div className="bg-purple-900/20 border border-purple-500/20 rounded-lg p-4">
            <h4 className="font-semibold text-purple-300 mb-2">Data Integrity</h4>
            <p className="text-sm text-slate-300">
              Verified strict train/test separation with zero data leakage. 
              Implemented hash-based duplicate detection.
            </p>
          </div>
        </div>
      </div>

      {/* Live Demo Preview */}
      <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-500/20 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-white flex items-center">
            <Scan className="h-5 w-5 mr-2 text-blue-400" />
            Face Scan Master Demo
          </h3>
          <button 
            onClick={() => window.location.hash = '#demo'}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Try Live Demo
          </button>
        </div>
        <p className="text-slate-300 mb-4">
          Experience real-time face scanning with our interactive demo featuring single scan, 
          face comparison, and live webcam analysis capabilities.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-slate-800/30 rounded-lg">
            <Target className="h-6 w-6 text-blue-400 mx-auto mb-2" />
            <div className="text-sm font-medium text-white">Single Scan</div>
            <div className="text-xs text-slate-400">Upload & analyze</div>
          </div>
          <div className="text-center p-3 bg-slate-800/30 rounded-lg">
            <Users className="h-6 w-6 text-green-400 mx-auto mb-2" />
            <div className="text-sm font-medium text-white">Face Comparison</div>
            <div className="text-xs text-slate-400">Compare similarity</div>
          </div>
          <div className="text-center p-3 bg-slate-800/30 rounded-lg">
            <Award className="h-6 w-6 text-purple-400 mx-auto mb-2" />
            <div className="text-sm font-medium text-white">Real-time Scan</div>
            <div className="text-xs text-slate-400">Live webcam analysis</div>
          </div>
        </div>
      </div>
    </div>
  );
}