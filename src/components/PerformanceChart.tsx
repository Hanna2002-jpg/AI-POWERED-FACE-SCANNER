import React from 'react';

export function PerformanceChart() {
  const data = [
    { epoch: 1, accuracy: 0.78, loss: 0.85 },
    { epoch: 5, accuracy: 0.84, loss: 0.65 },
    { epoch: 10, accuracy: 0.89, loss: 0.45 },
    { epoch: 15, accuracy: 0.92, loss: 0.35 },
    { epoch: 20, accuracy: 0.942, loss: 0.28 }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center p-4 bg-slate-700/30 rounded-lg">
          <div className="text-2xl font-bold text-green-400">94.2%</div>
          <div className="text-sm text-slate-300">Final Accuracy</div>
        </div>
        <div className="text-center p-4 bg-slate-700/30 rounded-lg">
          <div className="text-2xl font-bold text-blue-400">0.28</div>
          <div className="text-sm text-slate-300">Final Loss</div>
        </div>
      </div>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm text-slate-300 mb-2">
            <span>Training Progress</span>
            <span>20 Epochs</span>
          </div>
          <div className="relative h-32 bg-slate-700/30 rounded-lg overflow-hidden">
            <svg className="w-full h-full" viewBox="0 0 400 120">
              {/* Accuracy line */}
              <polyline
                points="20,100 100,75 180,45 260,25 380,15"
                fill="none"
                stroke="#22c55e"
                strokeWidth="2"
                className="drop-shadow-lg"
              />
              {/* Loss line */}
              <polyline
                points="20,15 100,35 180,55 260,65 380,72"
                fill="none"
                stroke="#ef4444"
                strokeWidth="2"
                className="drop-shadow-lg"
              />
              {/* Data points */}
              {data.map((point, idx) => (
                <g key={idx}>
                  <circle
                    cx={20 + (idx * 90)}
                    cy={120 - (point.accuracy * 100)}
                    r="3"
                    fill="#22c55e"
                  />
                  <circle
                    cx={20 + (idx * 90)}
                    cy={point.loss * 100}
                    r="3"
                    fill="#ef4444"
                  />
                </g>
              ))}
            </svg>
          </div>
          <div className="flex justify-between text-xs text-slate-400 mt-2">
            <span>Epoch 1</span>
            <span>Epoch 5</span>
            <span>Epoch 10</span>
            <span>Epoch 15</span>
            <span>Epoch 20</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            <span className="text-slate-300">Accuracy</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-400 rounded-full"></div>
            <span className="text-slate-300">Loss</span>
          </div>
        </div>
      </div>
    </div>
  );
}