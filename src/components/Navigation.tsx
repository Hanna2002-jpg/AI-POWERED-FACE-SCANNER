import React from 'react';
import { 
  BarChart3, 
  FolderTree, 
  Database, 
  Brain, 
  Target, 
  FileText,
  Home,
  Scan
} from 'lucide-react';

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Home },
  { id: 'demo', label: 'Face Scan Demo', icon: Scan },
  { id: 'structure', label: 'Project Structure', icon: FolderTree },
  { id: 'dataset', label: 'Dataset Manager', icon: Database },
  { id: 'training', label: 'Training Pipeline', icon: Brain },
  { id: 'evaluation', label: 'Model Evaluation', icon: Target },
  { id: 'report', label: 'Report Generator', icon: FileText }
];

export function Navigation({ activeTab, setActiveTab }: NavigationProps) {
  return (
    <nav className="bg-slate-800/50 backdrop-blur-xl border-b border-slate-700/50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <Scan className="h-8 w-8 text-blue-400" />
            <span className="text-xl font-bold text-white">Face Scan Master</span>
          </div>
          
          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    activeTab === item.id
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}