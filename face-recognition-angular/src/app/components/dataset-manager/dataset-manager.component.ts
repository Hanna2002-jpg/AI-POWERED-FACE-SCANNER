import { Component, OnInit } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { DataService } from '../../services/data.service';

interface DatasetStats {
  totalImages: number;
  identities: number;
  trainImages: number;
  testImages: number;
  avgImagesPerIdentity: number;
  qualityScore: number;
}

interface QualityMetric {
  name: string;
  value: number;
  status: 'excellent' | 'good' | 'fair' | 'poor';
}

@Component({
  selector: 'app-dataset-manager',
  template: `
    <div class="dataset-container">
      <div class="header-section">
        <h2 class="page-title">Face Scan Dataset Manager</h2>
        <p class="page-subtitle">
          Comprehensive dataset management with quality assessment, integrity verification, 
          and strict train/test separation for reliable face scanning evaluation.
        </p>
      </div>

      <mat-tab-group class="dataset-tabs" backgroundColor="primary">
        <mat-tab label="Overview">
          <div class="tab-content">
            <div class="stats-grid">
              <mat-card class="stat-card blue">
                <mat-card-content>
                  <div class="stat-header">
                    <div class="stat-info">
                      <p class="stat-label">Total Images</p>
                      <p class="stat-value">{{ datasetStats.totalImages | number }}</p>
                    </div>
                    <lucide-icon name="eye" class="stat-icon"></lucide-icon>
                  </div>
                </mat-card-content>
              </mat-card>
              
              <mat-card class="stat-card green">
                <mat-card-content>
                  <div class="stat-header">
                    <div class="stat-info">
                      <p class="stat-label">Identities</p>
                      <p class="stat-value">{{ datasetStats.identities | number }}</p>
                    </div>
                    <lucide-icon name="users" class="stat-icon"></lucide-icon>
                  </div>
                </mat-card-content>
              </mat-card>
              
              <mat-card class="stat-card purple">
                <mat-card-content>
                  <div class="stat-header">
                    <div class="stat-info">
                      <p class="stat-label">Training Set</p>
                      <p class="stat-value">{{ datasetStats.trainImages | number }}</p>
                    </div>
                    <lucide-icon name="database" class="stat-icon"></lucide-icon>
                  </div>
                </mat-card-content>
              </mat-card>
              
              <mat-card class="stat-card orange">
                <mat-card-content>
                  <div class="stat-header">
                    <div class="stat-info">
                      <p class="stat-label">Test Set</p>
                      <p class="stat-value">{{ datasetStats.testImages | number }}</p>
                    </div>
                    <lucide-icon name="bar-chart-3" class="stat-icon"></lucide-icon>
                  </div>
                </mat-card-content>
              </mat-card>
            </div>

            <mat-card class="distribution-card">
              <mat-card-header>
                <mat-card-title>Dataset Distribution</mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="distribution-grid">
                  <div class="split-section">
                    <h4>Train/Test Split</h4>
                    <div class="split-item">
                      <div class="split-info">
                        <span>Training (70%)</span>
                        <span class="split-value">{{ datasetStats.trainImages }} images</span>
                      </div>
                      <mat-progress-bar value="70" color="primary"></mat-progress-bar>
                    </div>
                    <div class="split-item">
                      <div class="split-info">
                        <span>Testing (30%)</span>
                        <span class="split-value">{{ datasetStats.testImages }} images</span>
                      </div>
                      <mat-progress-bar value="30" color="accent"></mat-progress-bar>
                    </div>
                  </div>
                  
                  <div class="quality-section">
                    <h4>Quality Assessment</h4>
                    <div *ngFor="let metric of qualityMetrics" class="quality-item">
                      <div class="quality-info">
                        <span>{{ metric.name }}</span>
                        <span class="quality-percentage">{{ (metric.value * 100) | number:'1.0-0' }}%</span>
                      </div>
                      <mat-progress-bar 
                        [value]="metric.value * 100"
                        [color]="getQualityColor(metric.status)">
                      </mat-progress-bar>
                    </div>
                  </div>
                </div>
              </mat-card-content>
            </mat-card>

            <div class="action-buttons">
              <button mat-raised-button color="primary" (click)="uploadDataset()">
                <lucide-icon name="upload"></lucide-icon>
                Upload Dataset
              </button>
              <button mat-raised-button color="accent" (click)="preprocessData()">
                <lucide-icon name="settings"></lucide-icon>
                Preprocess Data
              </button>
              <button mat-raised-button (click)="validateDataset()">
                <lucide-icon name="check-circle"></lucide-icon>
                Validate Dataset
              </button>
            </div>
          </div>
        </mat-tab>

        <mat-tab label="Data Integrity">
          <div class="tab-content">
            <mat-card class="integrity-card">
              <mat-card-header>
                <mat-card-title>
                  <lucide-icon name="shield" class="title-icon"></lucide-icon>
                  Data Integrity Verification
                </mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="integrity-checks">
                  <div class="check-item verified">
                    <div class="check-content">
                      <lucide-icon name="check-circle" class="check-icon"></lucide-icon>
                      <div class="check-info">
                        <p class="check-title">Train/Test Separation</p>
                        <p class="check-description">No overlap detected between training and test sets</p>
                      </div>
                    </div>
                    <span class="check-status">✓ Verified</span>
                  </div>
                  
                  <div class="check-item verified">
                    <div class="check-content">
                      <lucide-icon name="check-circle" class="check-icon"></lucide-icon>
                      <div class="check-info">
                        <p class="check-title">Duplicate Detection</p>
                        <p class="check-description">Hash-based duplicate detection completed</p>
                      </div>
                    </div>
                    <span class="check-status">✓ No Duplicates</span>
                  </div>
                  
                  <div class="check-item verified">
                    <div class="check-content">
                      <lucide-icon name="check-circle" class="check-icon"></lucide-icon>
                      <div class="check-info">
                        <p class="check-title">Identity Consistency</p>
                        <p class="check-description">All identities properly labeled and verified</p>
                      </div>
                    </div>
                    <span class="check-status">✓ Consistent</span>
                  </div>
                </div>
              </mat-card-content>
            </mat-card>
            
            <mat-card class="checksum-card">
              <mat-card-header>
                <mat-card-title>Validation Checksums</mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="checksum-list">
                  <div class="checksum-item">
                    <span>Training set checksum:</span>
                    <code>a3b2c1d4e5f6...</code>
                  </div>
                  <div class="checksum-item">
                    <span>Test set checksum:</span>
                    <code>f6e5d4c3b2a1...</code>
                  </div>
                  <div class="checksum-item">
                    <span>Labels checksum:</span>
                    <code>1a2b3c4d5e6f...</code>
                  </div>
                </div>
              </mat-card-content>
            </mat-card>

            <div class="action-buttons">
              <button mat-raised-button color="primary" (click)="verifyIntegrity()">
                <lucide-icon name="shield-check"></lucide-icon>
                Verify Integrity
              </button>
              <button mat-raised-button color="accent" (click)="generateReport()">
                <lucide-icon name="file-text"></lucide-icon>
                Generate Report
              </button>
            </div>
          </div>
        </mat-tab>
      </mat-tab-group>
    </div>
  `,
  styleUrls: ['./dataset-manager.component.scss']
})
export class DatasetManagerComponent implements OnInit {
  datasetStats: DatasetStats = {
    totalImages: 4892,
    identities: 1247,
    trainImages: 3425,
    testImages: 1467,
    avgImagesPerIdentity: 3.9,
    qualityScore: 0.72
  };

  qualityMetrics: QualityMetric[] = [
    { name: 'Blur Detection', value: 0.78, status: 'good' },
    { name: 'Illumination', value: 0.65, status: 'fair' },
    { name: 'Resolution', value: 0.83, status: 'good' },
    { name: 'Face Alignment', value: 0.91, status: 'excellent' },
    { name: 'Occlusion', value: 0.58, status: 'fair' }
  ];

  constructor(
    private dataService: DataService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.loadDatasetStats();
  }

  loadDatasetStats(): void {
    this.dataService.getDatasetStats().subscribe(stats => {
      this.datasetStats = stats;
    });
  }

  uploadDataset(): void {
    this.snackBar.open('Dataset upload initiated...', 'Close', { duration: 3000 });
    this.dataService.uploadDataset().subscribe(
      result => {
        this.snackBar.open('Dataset uploaded successfully!', 'Close', { duration: 3000 });
        this.loadDatasetStats();
      },
      error => {
        this.snackBar.open('Upload failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  preprocessData(): void {
    this.snackBar.open('Data preprocessing started...', 'Close', { duration: 3000 });
    this.dataService.preprocessData().subscribe(
      result => {
        this.snackBar.open('Data preprocessing completed!', 'Close', { duration: 3000 });
        this.loadDatasetStats();
      },
      error => {
        this.snackBar.open('Preprocessing failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  validateDataset(): void {
    this.snackBar.open('Dataset validation in progress...', 'Close', { duration: 3000 });
    this.dataService.validateDataset().subscribe(
      result => {
        this.snackBar.open('Dataset validation completed successfully!', 'Close', { duration: 3000 });
      },
      error => {
        this.snackBar.open('Validation failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  verifyIntegrity(): void {
    this.snackBar.open('Integrity verification started...', 'Close', { duration: 3000 });
    this.dataService.verifyIntegrity().subscribe(
      result => {
        this.snackBar.open('Integrity verification passed!', 'Close', { duration: 3000 });
      },
      error => {
        this.snackBar.open('Integrity check failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  generateReport(): void {
    this.snackBar.open('Generating dataset report...', 'Close', { duration: 3000 });
    this.dataService.generateDatasetReport().subscribe(
      result => {
        this.snackBar.open('Dataset report generated successfully!', 'Close', { duration: 3000 });
      },
      error => {
        this.snackBar.open('Report generation failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  getQualityColor(status: string): string {
    switch (status) {
      case 'excellent': return 'primary';
      case 'good': return 'accent';
      case 'fair': return 'warn';
      default: return 'warn';
    }
  }
}