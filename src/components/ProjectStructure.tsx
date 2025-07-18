import React from 'react';
import { FolderOpen, File, Code, Database, Brain, BarChart3, FileText, Settings } from 'lucide-react';

interface FileNodeProps {
  name: string;
  type: 'folder' | 'file';
  children?: FileNodeProps[];
  icon?: React.ElementType;
  description?: string;
}

function FileNode({ name, type, children, icon: Icon, description }: FileNodeProps) {
  const [isExpanded, setIsExpanded] = React.useState(true);
  
  return (
    <div className="ml-4">
      <div 
        className="flex items-center space-x-2 py-1 px-2 rounded-lg hover:bg-slate-700/30 cursor-pointer transition-colors"
        onClick={() => type === 'folder' && setIsExpanded(!isExpanded)}
      >
        {type === 'folder' ? (
          <FolderOpen className="h-4 w-4 text-blue-400" />
        ) : (
          Icon ? <Icon className="h-4 w-4 text-slate-400" /> : <File className="h-4 w-4 text-slate-400" />
        )}
        <span className={`${type === 'folder' ? 'text-blue-300 font-medium' : 'text-slate-300'}`}>
          {name}
        </span>
        {description && (
          <span className="text-xs text-slate-500 italic">- {description}</span>
        )}
      </div>
      
      {type === 'folder' && isExpanded && children && (
        <div className="ml-2 border-l border-slate-600/50 pl-2">
          {children.map((child, idx) => (
            <FileNode key={idx} {...child} />
          ))}
        </div>
      )}
    </div>
  );
}

export function ProjectStructure() {
  const projectStructure: FileNodeProps = {
    name: 'face-recognition-project',
    type: 'folder',
    children: [
      {
        name: 'src',
        type: 'folder',
        children: [
          { name: 'main.py', type: 'file', icon: Code, description: 'Main execution script with CLI' },
          { name: 'train.py', type: 'file', icon: Brain, description: 'Training pipeline implementation' },
          { name: 'config.py', type: 'file', icon: Settings, description: 'Configuration management' }
        ]
      },
      {
        name: 'data',
        type: 'folder',
        children: [
          { name: 'dataset_handler.py', type: 'file', icon: Database, description: 'Dataset loading and preprocessing' },
          { name: 'augmentation.py', type: 'file', icon: Database, description: 'Data augmentation utilities' },
          { name: 'quality_assessment.py', type: 'file', icon: BarChart3, description: 'Image quality analysis' }
        ]
      },
      {
        name: 'models',
        type: 'folder',
        children: [
          { name: 'base_model.py', type: 'file', icon: Brain, description: 'Abstract base class for models' },
          { name: 'arcface_model.py', type: 'file', icon: Brain, description: 'ArcFace implementation' },
          { name: 'insightface_model.py', type: 'file', icon: Brain, description: 'InsightFace implementation' },
          { name: 'model_utils.py', type: 'file', icon: Code, description: 'Model utilities and helpers' }
        ]
      },
      {
        name: 'evaluation',
        type: 'folder',
        children: [
          { name: 'baseline_evaluation.py', type: 'file', icon: BarChart3, description: 'Pre-training evaluation' },
          { name: 'metrics.py', type: 'file', icon: BarChart3, description: 'Evaluation metrics implementation' },
          { name: 'visualizations.py', type: 'file', icon: BarChart3, description: 'Plotting and visualization' },
          { name: 'post_training_eval.py', type: 'file', icon: BarChart3, description: 'Post-training evaluation' }
        ]
      },
      {
        name: 'reports',
        type: 'folder',
        children: [
          { name: 'report_generator.py', type: 'file', icon: FileText, description: 'Automated report generation' },
          { name: 'template.md', type: 'file', icon: FileText, description: 'Report template' },
          { name: 'figures', type: 'folder', children: [] }
        ]
      },
      {
        name: 'outputs',
        type: 'folder',
        children: [
          { name: 'checkpoints', type: 'folder', children: [] },
          { name: 'logs', type: 'folder', children: [] },
          { name: 'results', type: 'folder', children: [] }
        ]
      },
      { name: 'requirements.txt', type: 'file', icon: Settings, description: 'Python dependencies' },
      { name: 'README.md', type: 'file', icon: FileText, description: 'Setup and usage instructions' },
      { name: 'config.yaml', type: 'file', icon: Settings, description: 'Configuration file' }
    ]
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-4">Project Structure</h2>
        <p className="text-slate-300 max-w-2xl mx-auto">
          Complete implementation structure for face recognition model evaluation with 
          modular architecture and comprehensive documentation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <FolderOpen className="h-5 w-5 mr-2 text-blue-400" />
            Directory Structure
          </h3>
          <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm max-h-96 overflow-y-auto">
            <FileNode {...projectStructure} />
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-white mb-4">Implementation Features</h3>
          <div className="space-y-4">
            <div className="p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg">
              <h4 className="font-semibold text-blue-300 mb-2">Modular Architecture</h4>
              <p className="text-sm text-slate-300">
                Clean separation of concerns with dedicated modules for data handling, 
                model implementation, evaluation, and reporting.
              </p>
            </div>
            
            <div className="p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
              <h4 className="font-semibold text-green-300 mb-2">Quality Assurance</h4>
              <p className="text-sm text-slate-300">
                Comprehensive testing, type hints, documentation, and automated 
                data leakage prevention measures.
              </p>
            </div>
            
            <div className="p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
              <h4 className="font-semibold text-purple-300 mb-2">Reproducibility</h4>
              <p className="text-sm text-slate-300">
                Version control, random seed management, configuration files, 
                and environment specification for consistent results.
              </p>
            </div>
            
            <div className="p-4 bg-orange-900/20 border border-orange-500/20 rounded-lg">
              <h4 className="font-semibold text-orange-300 mb-2">Professional Output</h4>
              <p className="text-sm text-slate-300">
                Automated report generation, structured outputs, and 
                comprehensive documentation suitable for industry evaluation.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-white mb-4">Technical Stack</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <Brain className="h-8 w-8 text-blue-400 mx-auto mb-2" />
            <h4 className="font-semibold text-white mb-1">Deep Learning</h4>
            <p className="text-sm text-slate-300">PyTorch, TensorFlow, OpenCV</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <BarChart3 className="h-8 w-8 text-green-400 mx-auto mb-2" />
            <h4 className="font-semibold text-white mb-1">Analytics</h4>
            <p className="text-sm text-slate-300">scikit-learn, matplotlib, seaborn</p>
          </div>
          <div className="text-center p-4 bg-slate-700/30 rounded-lg">
            <FileText className="h-8 w-8 text-purple-400 mx-auto mb-2" />
            <h4 className="font-semibold text-white mb-1">Documentation</h4>
            <p className="text-sm text-slate-300">LaTeX, Markdown, Jupyter</p>
          </div>
        </div>
      </div>
    </div>
  );
}