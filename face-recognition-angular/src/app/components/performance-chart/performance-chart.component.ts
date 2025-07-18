import { Component, OnInit } from '@angular/core';
import { ChartConfiguration, ChartOptions, ChartType } from 'chart.js';

@Component({
  selector: 'app-performance-chart',
  template: `
    <div class="chart-container">
      <div class="chart-stats">
        <div class="stat-item">
          <div class="stat-value">94.2%</div>
          <div class="stat-label">Final Accuracy</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">0.28</div>
          <div class="stat-label">Final Loss</div>
        </div>
      </div>
      
      <div class="chart-wrapper">
        <canvas 
          baseChart
          [data]="lineChartData"
          [options]="lineChartOptions"
          [type]="lineChartType">
        </canvas>
      </div>
      
      <div class="chart-legend">
        <div class="legend-item">
          <div class="legend-color accuracy"></div>
          <span>Accuracy</span>
        </div>
        <div class="legend-item">
          <div class="legend-color loss"></div>
          <span>Loss</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .chart-container {
      padding: 1rem 0;
    }
    
    .chart-stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    
    .stat-item {
      text-align: center;
      padding: 1rem;
      background: rgba(71, 85, 105, 0.3);
      border-radius: 0.5rem;
    }
    
    .stat-value {
      font-size: 1.5rem;
      font-weight: 700;
      margin-bottom: 0.25rem;
    }
    
    .stat-value:first-child {
      color: #4ade80;
    }
    
    .stat-value:last-child {
      color: #60a5fa;
    }
    
    .stat-label {
      font-size: 0.875rem;
      color: #cbd5e1;
    }
    
    .chart-wrapper {
      height: 200px;
      margin-bottom: 1rem;
    }
    
    .chart-legend {
      display: flex;
      justify-content: center;
      gap: 1rem;
    }
    
    .legend-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: #cbd5e1;
    }
    
    .legend-color {
      width: 0.75rem;
      height: 0.75rem;
      border-radius: 50%;
    }
    
    .legend-color.accuracy {
      background-color: #4ade80;
    }
    
    .legend-color.loss {
      background-color: #f87171;
    }
  `]
})
export class PerformanceChartComponent implements OnInit {
  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: ['Epoch 1', 'Epoch 5', 'Epoch 10', 'Epoch 15', 'Epoch 20'],
    datasets: [
      {
        data: [78, 84, 89, 92, 94.2],
        label: 'Accuracy',
        fill: false,
        tension: 0.4,
        borderColor: '#4ade80',
        backgroundColor: '#4ade80',
        pointBackgroundColor: '#4ade80',
        pointBorderColor: '#4ade80',
        pointRadius: 4
      },
      {
        data: [85, 65, 45, 35, 28],
        label: 'Loss',
        fill: false,
        tension: 0.4,
        borderColor: '#f87171',
        backgroundColor: '#f87171',
        pointBackgroundColor: '#f87171',
        pointBorderColor: '#f87171',
        pointRadius: 4
      }
    ]
  };

  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(71, 85, 105, 0.3)'
        },
        ticks: {
          color: '#cbd5e1'
        }
      },
      y: {
        grid: {
          color: 'rgba(71, 85, 105, 0.3)'
        },
        ticks: {
          color: '#cbd5e1'
        }
      }
    }
  };

  public lineChartType: ChartType = 'line';

  ngOnInit(): void {
    // Initialize chart data
  }
}