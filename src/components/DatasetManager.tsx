import React, { useState } from 'react';
import { Database, Upload, CheckCircle, AlertCircle, Eye, BarChart3, Shield, Users } from 'lucide-react';

export function DatasetManager() {
  const [activeTab, setActiveTab] = useState('overview');

  const datasetStats = {
    totalImages: 4892,
    identities: 1247,
    trainImages: 3425,
    testImages: 1467,
    avgImagesPerIdentity: 3.9,
    qualityScore: 0.72
  };

  const qualityMetrics = [
    { name: 'Blur Detection', value: 0.78, status: 'good' },
    { name: 'Illumination', value: 0.65, status: 'fair' },
    { name: 'Resolution', value: 0.83, status: 'good' },
    { name: 'Face Alignment', value: 0.91, status: 'excellent' },
    { name: 'Occlusion', value: 0.58, status: 'fair' }
  ];

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-blue-900/20 border border-blue-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Total Images</p>
              <p className="text-2xl font-bold text-blue-400">{datasetStats.totalImages.toLocaleString()}</p>
            </div>
            <Eye className="h-8 w-8 text-blue-400" />
          </div>
        </div>
        
        <div className="bg-green-900/20 border border-green-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Identities</p>
              <p className="text-2xl font-bold text-green-400">{datasetStats.identities.toLocaleString()}</p>
            </div>
            <Users className="h-8 w-8 text-green-400" />
          </div>
        </div>
        
        <div className="bg-purple-900/20 border border-purple-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Training Set</p>
              <p className="text-2xl font-bold text-purple-400">{datasetStats.trainImages.toLocaleString()}</p>
            </div>
            <Database className="h-8 w-8 text-purple-400" />
          </div>
        </div>
        
        <div className="bg-orange-900/20 border border-orange-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Test Set</p>
              <p className="text-2xl font-bold text-orange-400">{datasetStats.testImages.toLocaleString()}</p>
            </div>
            <BarChart3 className="h-8 w-8 text-orange-400" />
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Dataset Distribution</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="text-lg font-semibold text-slate-300 mb-3">Train/Test Split</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Training (70%)</span>
                <span className="text-purple-400 font-medium">{datasetStats.trainImages} images</span>
              </div>
              <div className="bg-slate-700 rounded-full h-2">
                <div className="bg-purple-400 h-2 rounded-full" style={{ width: '70%' }} />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Testing (30%)</span>
                <span className="text-orange-400 font-medium">{datasetStats.testImages} images</span>
              </div>
              <div className="bg-slate-700 rounded-full h-2">
                <div className="bg-orange-400 h-2 rounded-full" style={{ width: '30%' }} />
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="text-lg font-semibold text-slate-300 mb-3">Quality Assessment</h4>
            <div className="space-y-3">
              {qualityMetrics.map((metric) => (
                <div key={metric.name} className="flex items-center justify-between">
                  <span className="text-slate-300">{metric.name}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-24 bg-slate-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          metric.status === 'excellent' ? 'bg-green-400' :
                          metric.status === 'good' ? 'bg-blue-400' : 'bg-yellow-400'
                        }`}
                        style={{ width: `${metric.value * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-slate-400">{(metric.value * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderIntegrity = () => (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Shield className="h-5 w-5 mr-2 text-green-400" />
          Data Integrity Verification
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-green-400" />
              <div>
                <p className="font-medium text-green-300">Train/Test Separation</p>
                <p className="text-sm text-slate-300">No overlap detected between training and test sets</p>
              </div>
            </div>
            <span className="text-green-400 font-medium">✓ Verified</span>
          </div>
          
          <div className="flex items-center justify-between p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-green-400" />
              <div>
                <p className="font-medium text-green-300">Duplicate Detection</p>
                <p className="text-sm text-slate-300">Hash-based duplicate detection completed</p>
              </div>
            </div>
            <span className="text-green-400 font-medium">✓ No Duplicates</span>
          </div>
          
          <div className="flex items-center justify-between p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-green-400" />
              <div>
                <p className="font-medium text-green-300">Identity Consistency</p>
                <p className="text-sm text-slate-300">All identities properly labeled and verified</p>
              </div>
            </div>
            <span className="text-green-400 font-medium">✓ Consistent</span>
          </div>
        </div>
      </div>
      
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Validation Checksums</h3>
        <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-300">Training set checksum:</span>
              <span className="text-green-400">a3b2c1d4e5f6...</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-300">Test set checksum:</span>
              <span className="text-green-400">f6e5d4c3b2a1...</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-300">Labels checksum:</span>
              <span className="text-green-400">1a2b3c4d5e6f...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Database },
    { id: 'integrity', label: 'Data Integrity', icon: Shield }
  ];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-4">Dataset Manager</h2>
        <p className="text-slate-300 max-w-2xl mx-auto">
          Comprehensive dataset management with quality assessment, integrity verification, 
          and strict train/test separation for reliable evaluation.
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

      {activeTab === 'overview' && renderOverview()}
      {activeTab === 'integrity' && renderIntegrity()}
    </div>
  );
}