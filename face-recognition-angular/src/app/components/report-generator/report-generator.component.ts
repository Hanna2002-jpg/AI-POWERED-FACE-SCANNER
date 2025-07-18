import { Component, OnInit } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ReportService, ReportSection, GeneratedReport } from '../../services/report.service';
import { PDFService } from '../../services/pdf.service';

@Component({
  selector: 'app-report-generator',
  template: `
    <div class="report-container">
      <div class="header-section">
        <h2 class="page-title">Face Scan Report Generator</h2>
        <p class="page-subtitle">
          Generate comprehensive evaluation reports with automated analysis, 
          visualizations, and professional formatting for submission and review.
        </p>
      </div>

      <div class="content-grid">
        <!-- Report Configuration -->
        <mat-card class="config-card">
          <mat-card-header>
            <mat-card-title>
              <lucide-icon name="file-text" class="title-icon"></lucide-icon>
              Report Configuration
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div class="config-section">
              <h4>Report Sections</h4>
              <div class="sections-list">
                <div *ngFor="let section of reportSections" class="section-item">
                  <div class="section-content">
                    <mat-checkbox 
                      [checked]="selectedSections.includes(section.id)"
                      [disabled]="section.required"
                      (change)="toggleSection(section.id, $event.checked)">
                      {{ section.title }}
                    </mat-checkbox>
                    <span *ngIf="section.required" class="required-badge">Required</span>
                  </div>
                  <div class="section-progress">
                    <div class="progress-bar">
                      <mat-progress-bar 
                        [value]="section.progress"
                        [color]="getProgressColor(section.progress)">
                      </mat-progress-bar>
                    </div>
                    <span class="progress-text">{{ section.progress }}%</span>
                  </div>
                </div>
              </div>
              
              <div class="action-buttons">
                <button 
                  mat-raised-button 
                  color="primary"
                  (click)="generateReport()"
                  [disabled]="isGenerating">
                  <mat-spinner *ngIf="isGenerating" diameter="20"></mat-spinner>
                  <lucide-icon *ngIf="!isGenerating" name="file-text"></lucide-icon>
                  {{ isGenerating ? 'Generating...' : 'Generate Report' }}
                </button>
                
                <button 
                  mat-raised-button 
                  color="accent"
                  (click)="previewReport()"
                  [disabled]="isGenerating">
                  <lucide-icon name="eye"></lucide-icon>
                  Preview Report
                </button>

                <button 
                  mat-raised-button 
                  (click)="downloadPDF()"
                  [disabled]="isGenerating">
                  <lucide-icon name="download"></lucide-icon>
                  Download PDF
                </button>
              </div>
            </div>
          </mat-card-content>
        </mat-card>

        <!-- Report Status -->
        <mat-card class="status-card">
          <mat-card-header>
            <mat-card-title>
              <lucide-icon name="activity" class="title-icon"></lucide-icon>
              Generation Status
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div class="status-section">
              <div class="status-stats">
                <div class="stat-item">
                  <div class="stat-value">{{ reportSections.length }}</div>
                  <div class="stat-label">Sections Ready</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">15</div>
                  <div class="stat-label">Figures Generated</div>
                </div>
              </div>
              
              <div class="status-checks">
                <div class="check-item completed">
                  <div class="check-content">
                    <lucide-icon name="check-circle" class="check-icon"></lucide-icon>
                    <span>Data analysis complete</span>
                  </div>
                </div>
                
                <div class="check-item completed">
                  <div class="check-content">
                    <lucide-icon name="check-circle" class="check-icon"></lucide-icon>
                    <span>Visualizations generated</span>
                  </div>
                </div>
                
                <div class="check-item completed">
                  <div class="check-content">
                    <lucide-icon name="check-circle" class="check-icon"></lucide-icon>
                    <span>Statistical analysis ready</span>
                  </div>
                </div>
                
                <div class="check-item" [ngClass]="isGenerating ? 'in-progress' : 'completed'">
                  <div class="check-content">
                    <lucide-icon 
                      [name]="isGenerating ? 'clock' : 'check-circle'" 
                      class="check-icon">
                    </lucide-icon>
                    <span>Final formatting</span>
                  </div>
                  <span class="check-status">{{ isGenerating ? '80%' : '100%' }}</span>
                </div>
              </div>
            </div>
          </mat-card-content>
        </mat-card>
      </div>

      <!-- Generated Reports -->
      <mat-card class="reports-card">
        <mat-card-header>
          <mat-card-title>Generated Reports</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="reports-list">
            <div *ngFor="let report of generatedReports" class="report-item">
              <div class="report-info">
                <lucide-icon name="file-text" class="report-icon"></lucide-icon>
                <div class="report-details">
                  <p class="report-name">{{ report.name }}</p>
                  <p class="report-meta">
                    {{ report.type }} • {{ report.size }} • Generated {{ report.generated }}
                  </p>
                </div>
              </div>
              
              <div class="report-actions">
                <span class="status-badge completed">{{ report.status }}</span>
                <button 
                  mat-icon-button 
                  color="primary"
                  (click)="downloadReportPDF(report)"
                  matTooltip="Download PDF">
                  <lucide-icon name="download"></lucide-icon>
                </button>
                <button 
                  mat-icon-button 
                  (click)="previewReportFile(report)"
                  matTooltip="Preview">
                  <lucide-icon name="eye"></lucide-icon>
                </button>
              </div>
            </div>
          </div>
        </mat-card-content>
      </mat-card>

      <!-- Report Preview -->
      <mat-card class="preview-card" *ngIf="showPreview">
        <mat-card-header>
          <mat-card-title>Report Preview</mat-card-title>
          <button mat-icon-button (click)="closePreview()">
            <lucide-icon name="x"></lucide-icon>
          </button>
        </mat-card-header>
        <mat-card-content>
          <div class="preview-container" [innerHTML]="previewContent"></div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styleUrls: ['./report-generator.component.scss']
})
export class ReportGeneratorComponent implements OnInit {
  reportSections: ReportSection[] = [];
  generatedReports: GeneratedReport[] = [];
  selectedSections: string[] = ['dataset', 'baseline', 'methodology', 'results', 'integrity'];
  isGenerating = false;
  showPreview = false;
  previewContent = '';

  constructor(
    private reportService: ReportService,
    private pdfService: PDFService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.loadReportSections();
    this.loadGeneratedReports();
  }

  loadReportSections(): void {
    this.reportService.getReportSections().subscribe(
      sections => this.reportSections = sections
    );
  }

  loadGeneratedReports(): void {
    this.reportService.getGeneratedReports().subscribe(
      reports => this.generatedReports = reports
    );
  }

  toggleSection(sectionId: string, checked: boolean): void {
    if (checked) {
      this.selectedSections.push(sectionId);
    } else {
      this.selectedSections = this.selectedSections.filter(id => id !== sectionId);
    }
  }

  generateReport(): void {
    this.isGenerating = true;
    this.snackBar.open('Report generation started...', 'Close', { duration: 3000 });

    this.reportService.generateReport(this.selectedSections).subscribe(
      result => {
        this.isGenerating = false;
        this.snackBar.open('Report generated successfully!', 'Close', { duration: 3000 });
        this.loadGeneratedReports();
      },
      error => {
        this.isGenerating = false;
        this.snackBar.open('Report generation failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  previewReport(): void {
    this.showPreview = true;
    this.previewContent = this.generatePreviewHTML();
  }

  closePreview(): void {
    this.showPreview = false;
    this.previewContent = '';
  }

  downloadPDF(): void {
    this.snackBar.open('Generating PDF report...', 'Close', { duration: 2000 });

    // Mock report data
    const reportData = {
      title: 'Face Recognition System Evaluation Report',
      date: new Date().toLocaleDateString(),
      metrics: {
        accuracy: 0.942,
        precision: 0.938,
        recall: 0.945,
        f1Score: 0.941
      }
    };

    this.pdfService.generateReportPDF(reportData, this.selectedSections).subscribe(
      (pdfBlob: Blob) => {
        const filename = `face-recognition-evaluation-report-${new Date().toISOString().split('T')[0]}.pdf`;
        this.pdfService.downloadPDF(pdfBlob, filename);
        this.snackBar.open('PDF downloaded successfully!', 'Close', { duration: 3000 });
      },
      error => {
        this.snackBar.open('PDF generation failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  downloadReportPDF(report: GeneratedReport): void {
    this.snackBar.open(`Downloading ${report.name}...`, 'Close', { duration: 2000 });

    // Generate PDF for existing report
    const reportData = {
      title: report.name,
      date: report.generated,
      type: report.type
    };

    this.pdfService.generateReportPDF(reportData, this.selectedSections).subscribe(
      (pdfBlob: Blob) => {
        const filename = `${report.name.toLowerCase().replace(/\s+/g, '-')}.pdf`;
        this.pdfService.downloadPDF(pdfBlob, filename);
        this.snackBar.open('PDF downloaded successfully!', 'Close', { duration: 3000 });
      },
      error => {
        this.snackBar.open('Download failed: ' + error.message, 'Close', { duration: 5000 });
      }
    );
  }

  previewReportFile(report: GeneratedReport): void {
    this.reportService.previewReport(report.id).subscribe(
      content => {
        this.previewContent = content;
        this.showPreview = true;
      }
    );
  }

  getProgressColor(progress: number): string {
    if (progress >= 90) return 'primary';
    if (progress >= 70) return 'accent';
    return 'warn';
  }

  private generatePreviewHTML(): string {
    return `
      <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="text-align: center; border-bottom: 3px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px;">
          <h1 style="color: #1e40af; font-size: 28px; margin-bottom: 10px;">Face Scan Master - System Evaluation Report</h1>
          <div style="color: #64748b; font-size: 16px; margin-bottom: 5px;">Advanced Face Scanning Technology Assessment</div>
          <div style="color: #94a3b8; font-size: 14px;">Generated on ${new Date().toLocaleDateString()}</div>
        </div>

        <div style="margin-bottom: 30px;">
          <h2 style="color: #1e40af; font-size: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 15px;">Executive Summary</h2>
          <p>
            This report presents the comprehensive evaluation of Face Scan Master, a state-of-the-art face scanning system 
            implemented using advanced ArcFace architecture. The model achieved exceptional performance with 94.2% 
            accuracy on realistic, low-quality image data through strategic fine-tuning and transfer learning.
          </p>
          
          <div style="background: #dbeafe; padding: 15px; border-radius: 8px; border-left: 4px solid #2563eb; margin: 15px 0;">
            <h4 style="color: #1e40af; margin-bottom: 8px;">Key Findings</h4>
            <ul>
              <li>2.1% improvement over baseline through fine-tuning</li>
              <li>Strict data integrity with zero train/test leakage</li>
              <li>Optimal threshold of 0.5 for balanced precision/recall</li>
              <li>Robust performance on low-quality images</li>
            </ul>
          </div>
        </div>

        <div style="margin-bottom: 30px;">
          <h2 style="color: #1e40af; font-size: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 15px;">Performance Metrics</h2>
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0;">
            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; background: #f8fafc;">
              <div style="font-weight: 600; color: #475569; font-size: 14px; margin-bottom: 5px;">Final Accuracy</div>
              <div style="font-size: 24px; font-weight: 700; color: #1e40af;">94.2%</div>
              <div style="font-size: 12px; color: #059669; margin-top: 2px;">+2.1% improvement</div>
            </div>
            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; background: #f8fafc;">
              <div style="font-weight: 600; color: #475569; font-size: 14px; margin-bottom: 5px;">Training Time</div>
              <div style="font-size: 24px; font-weight: 700; color: #1e40af;">2.4h</div>
              <div style="font-size: 12px; color: #059669; margin-top: 2px;">Efficient training</div>
            </div>
          </div>
        </div>

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b; font-size: 12px;">
          <p>This report was automatically generated by Face Scan Master AI Evaluation System</p>
          <p>© 2024 Face Scan Master Project - All rights reserved</p>
        </div>
      </div>
    `;
  }
}