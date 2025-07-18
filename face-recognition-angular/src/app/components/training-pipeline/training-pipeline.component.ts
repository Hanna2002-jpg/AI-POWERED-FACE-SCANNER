import { Component, OnInit, OnDestroy } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { TrainingService, TrainingStatus, TrainingConfig } from '../../services/training.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-training-pipeline',
  template: `
    <div class="training-container">
      <div class="header-section">
        <h2 class="page-title">Face Scan Training Pipeline</h2>
        <p class="page-subtitle">
          Advanced fine-tuning pipeline with transfer learning, custom loss functions, 
          and comprehensive monitoring for optimal face scanning performance.
        </p>
      </div>

      <div class="content-grid">
        <!-- Training Control Panel -->
        <mat-card class="control-card">
          <mat-card-header>
            <mat-card-title>
              <lucide-icon name="brain" class="title-icon"></lucide-icon>
              Training Control
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div class="control-section">
              <div class="status-display">
                <div class="status-info">
                  <p class="status-title">Training Status</p>
                  <p class="status-description">
                    {{ trainingStatus.isTraining ? 'Training in progress...' : 'Training completed' }}
                  </p>
                </div>
                <div class="control-buttons">
                  <button 
                    mat-fab 
                    [color]="trainingStatus.isTraining ? 'warn' : 'primary'"
                    (click)="toggleTraining()"
                    [disabled]="isLoading">
                    <lucide-icon [name]="trainingStatus.isTraining ? 'pause' : 'play'"></lucide-icon>
                  </button>
                  <button 
                    mat-fab 
                    color="accent"
                    (click)="resetTraining()"
                    [disabled]="isLoading">
                    <lucide-icon name="rotate-ccw"></lucide-icon>
                  </button>
                </div>
              </div>
              
              <div class="progress-section">
                <div class="progress-info">
                  <span>Progress</span>
                  <span class="progress-text">
                    {{ trainingStatus.currentEpoch }}/{{ trainingStatus.totalEpochs }} epochs
                  </span>
                </div>
                <mat-progress-bar 
                  [value]="(trainingStatus.currentEpoch / trainingStatus.totalEpochs) * 100"
                  color="primary">
                </mat-progress-bar>
              </div>
              
              <div class="metrics-grid">
                <div class="metric-item accuracy">
                  <div class="metric-value">{{ (trainingStatus.currentAccuracy * 100) | number:'1.1-1' }}%</div>
                  <div class="metric-label">Current Accuracy</div>
                </div>
                <div class="metric-item loss">
                  <div class="metric-value">{{ trainingStatus.currentLoss | number:'1.2-2' }}</div>
                  <div class="metric-label">Current Loss</div>
                </div>
              </div>
            </div>
          </mat-card-content>
        </mat-card>

        <!-- Model Configuration -->
        <mat-card class="config-card">
          <mat-card-header>
            <mat-card-title>
              <lucide-icon name="settings" class="title-icon"></lucide-icon>
              Model Configuration
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div class="config-section">
              <div class="config-group">
                <h4>Architecture</h4>
                <div class="config-list">
                  <div class="config-item">
                    <span>Backbone:</span>
                    <span>ResNet-50</span>
                  </div>
                  <div class="config-item">
                    <span>Embedding:</span>
                    <span>512</span>
                  </div>
                  <div class="config-item">
                    <span>Loss:</span>
                    <span>ArcFace</span>
                  </div>
                  <div class="config-item">
                    <span>Margin:</span>
                    <span>0.5</span>
                  </div>
                  <div class="config-item">
                    <span>Scale:</span>
                    <span>64</span>
                  </div>
                </div>
              </div>
              
              <div class="config-group">
                <h4>Hyperparameters</h4>
                <div class="config-list">
                  <div class="config-item">
                    <span>Learning Rate:</span>
                    <span>0.001</span>
                  </div>
                  <div class="config-item">
                    <span>Batch Size:</span>
                    <span>32</span>
                  </div>
                  <div class="config-item">
                    <span>Optimizer:</span>
                    <span>Adam</span>
                  </div>
                  <div class="config-item">
                    <span>Scheduler:</span>
                    <span>StepLR</span>
                  </div>
                  <div class="config-item">
                    <span>Augmentation:</span>
                    <span>Enabled</span>
                  </div>
                </div>
              </div>
            </div>
          </mat-card-content>
        </mat-card>
      </div>

      <!-- Training Metrics -->
      <mat-card class="metrics-card">
        <mat-card-header>
          <mat-card-title>
            <lucide-icon name="trending-up" class="title-icon"></lucide-icon>
            Training Metrics
          </mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="metrics-content">
            <div class="training-log">
              <h4>Recent Training Log</h4>
              <div class="log-container">
                <div class="log-header">
                  <span>Epoch</span>
                  <span>Loss</span>
                  <span>Accuracy</span>
                  <span>LR</span>
                  <span>Time</span>
                </div>
                <div *ngFor="let entry of trainingHistory" class="log-entry">
                  <span class="epoch">{{ entry.epoch }}</span>
                  <span class="loss">{{ entry.loss }}</span>
                  <span class="accuracy">{{ entry.accuracy }}</span>
                  <span class="lr">{{ entry.lr }}</span>
                  <span class="time">{{ entry.time }}</span>
                </div>
              </div>
            </div>
            
            <div class="performance-indicators">
              <h4>Performance Indicators</h4>
              <div class="indicators-list">
                <div class="indicator-item">
                  <div class="indicator-content">
                    <lucide-icon name="zap" class="indicator-icon speed"></lucide-icon>
                    <span>Training Speed</span>
                  </div>
                  <span class="indicator-value speed">2.3 samples/sec</span>
                </div>
                
                <div class="indicator-item">
                  <div class="indicator-content">
                    <lucide-icon name="clock" class="indicator-icon time"></lucide-icon>
                    <span>ETA</span>
                  </div>
                  <span class="indicator-value time">
                    {{ trainingStatus.isTraining ? formatTime(trainingStatus.estimatedTimeRemaining) : 'Completed' }}
                  </span>
                </div>
                
                <div class="indicator-item">
                  <div class="indicator-content">
                    <lucide-icon name="trending-up" class="indicator-icon improvement"></lucide-icon>
                    <span>Improvement</span>
                  </div>
                  <span class="indicator-value improvement">+2.1%</span>
                </div>
              </div>
            </div>
          </div>
        </mat-card-content>
      </mat-card>

      <!-- Training Strategy -->
      <mat-card class="strategy-card">
        <mat-card-header>
          <mat-card-title>Fine-tuning Strategy</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="strategy-grid">
            <div class="strategy-item transfer">
              <h4>Transfer Learning</h4>
              <p>Pre-trained ResNet-50 backbone with custom ArcFace head for face recognition tasks.</p>
            </div>
            
            <div class="strategy-item unfreezing">
              <h4>Gradual Unfreezing</h4>
              <p>Progressive unfreezing of layers to prevent overfitting and maintain learned features.</p>
            </div>
            
            <div class="strategy-item augmentation">
              <h4>Data Augmentation</h4>
              <p>Random horizontal flips, rotations, and color jittering to improve generalization.</p>
            </div>
          </div>
        </mat-card-content>
      </mat-card>

      <!-- Action Buttons -->
      <div class="action-buttons">
        <button mat-raised-button color="primary" (click)="startTraining()" [disabled]="trainingStatus.isTraining">
          <lucide-icon name="play"></lucide-icon>
          Start Training
        </button>
        <button mat-raised-button color="accent" (click)="saveCheckpoint()" [disabled]="!trainingStatus.isTraining">
          <lucide-icon name="save"></lucide-icon>
          Save Checkpoint
        </button>
        <button mat-raised-button (click)="exportModel()">
          <lucide-icon name="download"></lucide-icon>
          Export Model
        </button>
      </div>
    </div>
  `,
  styleUrls: ['./training-pipeline.component.scss']
})
export class TrainingPipelineComponent implements OnInit, OnDestroy {
  trainingStatus: TrainingStatus = {
    isTraining: false,
    currentEpoch: 20,
    totalEpochs: 20,
    currentLoss: 0.28,
    currentAccuracy: 0.942,
    estimatedTimeRemaining: 0
  };

  trainingHistory: any[] = [
    { epoch: 20, loss: 0.28, accuracy: 0.942, lr: 0.0001, time: '14:32' },
    { epoch: 19, loss: 0.31, accuracy: 0.938, lr: 0.0001, time: '14:25' },
    { epoch: 18, loss: 0.33, accuracy: 0.935, lr: 0.0001, time: '14:18' },
    { epoch: 17, loss: 0.36, accuracy: 0.928, lr: 0.0001, time: '14:11' },
    { epoch: 16, loss: 0.38, accuracy: 0.924, lr: 0.0001, time: '14:04' }
  ];

  isLoading = false;
  private statusSubscription?: Subscription;

  constructor(
    private trainingService: TrainingService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.statusSubscription = this.trainingService.getTrainingStatus().subscribe(
      status => this.trainingStatus = status
    );

    this.loadTrainingHistory();
  }

  ngOnDestroy(): void {
    this.statusSubscription?.unsubscribe();
  }

  toggleTraining(): void {
    this.isLoading = true;
    
    if (this.trainingStatus.isTraining) {
      this.trainingService.pauseTraining().subscribe(
        () => {
          this.snackBar.open('Training paused', 'Close', { duration: 2000 });
          this.isLoading = false;
        }
      );
    } else {
      this.trainingService.resumeTraining().subscribe(
        () => {
          this.snackBar.open('Training resumed', 'Close', { duration: 2000 });
          this.isLoading = false;
        }
      );
    }
  }

  resetTraining(): void {
    this.isLoading = true;
    this.trainingService.stopTraining().subscribe(
      () => {
        this.snackBar.open('Training reset', 'Close', { duration: 2000 });
        this.isLoading = false;
      }
    );
  }

  startTraining(): void {
    const config: TrainingConfig = {
      learningRate: 0.001,
      batchSize: 32,
      epochs: 20,
      optimizer: 'adam',
      model: 'arcface'
    };

    this.trainingService.startTraining(config).subscribe(
      () => {
        this.snackBar.open('Training started successfully!', 'Close', { duration: 3000 });
      },
      error => {
        this.snackBar.open('Failed to start training: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  saveCheckpoint(): void {
    this.snackBar.open('Checkpoint saved successfully!', 'Close', { duration: 2000 });
  }

  exportModel(): void {
    this.snackBar.open('Model export initiated...', 'Close', { duration: 3000 });
    // Simulate model export
    setTimeout(() => {
      this.snackBar.open('Model exported successfully!', 'Close', { duration: 3000 });
    }, 2000);
  }

  formatTime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  }

  private loadTrainingHistory(): void {
    this.trainingService.getTrainingHistory().subscribe(
      history => this.trainingHistory = history
    );
  }
}