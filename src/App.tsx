import React, { useState } from 'react';
import { Dashboard } from './components/Dashboard';
import { ProjectStructure } from './components/ProjectStructure';
import { ModelEvaluation } from './components/ModelEvaluation';
import { TrainingPipeline } from './components/TrainingPipeline';
import { ReportGenerator } from './components/ReportGenerator';
import { DatasetManager } from './components/DatasetManager';
import { Navigation } from './components/Navigation';
import { FaceScanDemo } from './components/FaceScanDemo';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'structure':
        return <ProjectStructure />;
      case 'dataset':
        return <DatasetManager />;
      case 'training':
        return <TrainingPipeline />;
      case 'evaluation':
        return <ModelEvaluation />;
      case 'report':
        return <ReportGenerator />;
      case 'demo':
        return <FaceScanDemo />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="container mx-auto px-4 py-8">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;