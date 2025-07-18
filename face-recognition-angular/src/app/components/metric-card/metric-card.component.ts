import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-metric-card',
  template: `
    <mat-card class="metric-card" [ngClass]="'metric-' + color">
      <mat-card-content>
        <div class="metric-header">
          <div class="metric-icon-container">
            <lucide-icon [name]="icon" class="metric-icon"></lucide-icon>
          </div>
          <div class="metric-change" [ngClass]="changeClass">
            {{ change }}
          </div>
        </div>
        <div class="metric-content">
          <h3 class="metric-title">{{ title }}</h3>
          <p class="metric-value">{{ value }}</p>
        </div>
      </mat-card-content>
    </mat-card>
  `,
  styles: [`
    .metric-card {
      background: rgba(30, 41, 59, 0.5);
      backdrop-filter: blur(12px);
      border: 1px solid;
      border-radius: 1rem;
      transition: all 0.3s ease;
      cursor: pointer;
    }
    
    .metric-card:hover {
      transform: scale(1.05);
    }
    
    .metric-blue {
      border-color: rgba(59, 130, 246, 0.2);
      background: rgba(59, 130, 246, 0.1);
    }
    
    .metric-green {
      border-color: rgba(34, 197, 94, 0.2);
      background: rgba(34, 197, 94, 0.1);
    }
    
    .metric-purple {
      border-color: rgba(168, 85, 247, 0.2);
      background: rgba(168, 85, 247, 0.1);
    }
    
    .metric-orange {
      border-color: rgba(249, 115, 22, 0.2);
      background: rgba(249, 115, 22, 0.1);
    }
    
    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }
    
    .metric-icon-container {
      padding: 0.75rem;
      border-radius: 0.75rem;
      border: 1px solid;
    }
    
    .metric-blue .metric-icon-container {
      background: rgba(59, 130, 246, 0.1);
      border-color: rgba(59, 130, 246, 0.2);
    }
    
    .metric-green .metric-icon-container {
      background: rgba(34, 197, 94, 0.1);
      border-color: rgba(34, 197, 94, 0.2);
    }
    
    .metric-purple .metric-icon-container {
      background: rgba(168, 85, 247, 0.1);
      border-color: rgba(168, 85, 247, 0.2);
    }
    
    .metric-orange .metric-icon-container {
      background: rgba(249, 115, 22, 0.1);
      border-color: rgba(249, 115, 22, 0.2);
    }
    
    .metric-icon {
      width: 1.5rem;
      height: 1.5rem;
    }
    
    .metric-blue .metric-icon {
      color: #60a5fa;
    }
    
    .metric-green .metric-icon {
      color: #4ade80;
    }
    
    .metric-purple .metric-icon {
      color: #a78bfa;
    }
    
    .metric-orange .metric-icon {
      color: #fb923c;
    }
    
    .metric-change {
      font-size: 0.875rem;
      font-weight: 500;
    }
    
    .positive {
      color: #4ade80;
    }
    
    .negative {
      color: #f87171;
    }
    
    .metric-title {
      color: #cbd5e1;
      font-size: 0.875rem;
      font-weight: 500;
      margin-bottom: 0.25rem;
    }
    
    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      margin: 0;
    }
    
    .metric-blue .metric-value {
      color: #60a5fa;
    }
    
    .metric-green .metric-value {
      color: #4ade80;
    }
    
    .metric-purple .metric-value {
      color: #a78bfa;
    }
    
    .metric-orange .metric-value {
      color: #fb923c;
    }
  `]
})
export class MetricCardComponent {
  @Input() title!: string;
  @Input() value!: string;
  @Input() change!: string;
  @Input() icon!: string;
  @Input() color!: string;

  get changeClass(): string {
    return this.change.startsWith('+') ? 'positive' : 'negative';
  }
}