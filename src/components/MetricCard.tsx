import React from 'react';
import { DivideIcon as LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string;
  change: string;
  icon: LucideIcon;
  color: 'blue' | 'green' | 'purple' | 'orange';
}

const colorMap = {
  blue: {
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
    text: 'text-blue-400',
    change: 'text-blue-300'
  },
  green: {
    bg: 'bg-green-500/10',
    border: 'border-green-500/20',
    text: 'text-green-400',
    change: 'text-green-300'
  },
  purple: {
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/20',
    text: 'text-purple-400',
    change: 'text-purple-300'
  },
  orange: {
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/20',
    text: 'text-orange-400',
    change: 'text-orange-300'
  }
};

export function MetricCard({ title, value, change, icon: Icon, color }: MetricCardProps) {
  const colors = colorMap[color];
  const isPositive = change.startsWith('+');
  
  return (
    <div className={`${colors.bg} ${colors.border} border backdrop-blur-xl rounded-2xl p-6 transition-all duration-300 hover:scale-105`}>
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-xl ${colors.bg} ${colors.border} border`}>
          <Icon className={`h-6 w-6 ${colors.text}`} />
        </div>
        <div className={`text-right ${colors.change}`}>
          <span className={`text-sm font-medium ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {change}
          </span>
        </div>
      </div>
      <div>
        <h3 className="text-slate-300 text-sm font-medium mb-1">{title}</h3>
        <p className={`text-2xl font-bold ${colors.text}`}>{value}</p>
      </div>
    </div>
  );
}