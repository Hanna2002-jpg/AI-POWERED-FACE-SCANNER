import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, delay } from 'rxjs';

export interface ReportSection {
  id: string;
  title: string;
  required: boolean;
  progress: number;
}

export interface GeneratedReport {
  id: number;
  name: string;
  type: string;
  size: string;
  generated: string;
  status: string;
  downloadUrl?: string;
}

@Injectable({
  providedIn: 'root'
})
export class ReportService {
  private apiUrl = '/api/reports';

  constructor(private http: HttpClient) {}

  getReportSections(): Observable<ReportSection[]> {
    const mockSections: ReportSection[] = [
      { id: 'dataset', title: 'Dataset Description and Rationale', required: true, progress: 100 },
      { id: 'baseline', title: 'Baseline Performance Analysis', required: true, progress: 100 },
      { id: 'methodology', title: 'Fine-tuning Methodology', required: true, progress: 100 },
      { id: 'results', title: 'Post-fine-tuning Results', required: true, progress: 100 },
      { id: 'integrity', title: 'Data Integrity Verification', required: true, progress: 100 },
      { id: 'appendix', title: 'Technical Appendix', required: false, progress: 80 },
      { id: 'samples', title: 'Sample Outputs', required: false, progress: 90 }
    ];

    return of(mockSections).pipe(delay(500));
  }

  getGeneratedReports(): Observable<GeneratedReport[]> {
    const mockReports: GeneratedReport[] = [
      {
        id: 1,
        name: 'Face Scan Master Evaluation Report',
        type: 'PDF',
        size: '2.4 MB',
        generated: '2024-01-15 14:30',
        status: 'completed',
        downloadUrl: '/reports/face-scan-master-evaluation.pdf'
      },
      {
        id: 2,
        name: 'Face Scan Master Technical Documentation',
        type: 'HTML',
        size: '1.8 MB',
        generated: '2024-01-15 14:25',
        status: 'completed',
        downloadUrl: '/reports/technical-documentation.html'
      },
      {
        id: 3,
        name: 'Face Scan Master Executive Summary',
        type: 'PDF',
        size: '0.9 MB',
        generated: '2024-01-15 14:20',
        status: 'completed',
        downloadUrl: '/reports/executive-summary.pdf'
      }
    ];

    return of(mockReports).pipe(delay(1000));
  }

  generateReport(sections: string[]): Observable<any> {
    return of({ 
      success: true, 
      message: 'Report generation started',
      reportId: Math.floor(Math.random() * 1000)
    }).pipe(delay(2000));
  }

  downloadReport(reportId: number): Observable<Blob> {
    // Simulate file download
    const mockPdfContent = new Blob(['Mock PDF content'], { type: 'application/pdf' });
    return of(mockPdfContent).pipe(delay(1000));
  }

  previewReport(reportId: number): Observable<string> {
    const mockPreview = `
      <h1>Face Scan Master - System Evaluation Report</h1>
      <h2>Executive Summary</h2>
      <p>This report presents comprehensive evaluation results...</p>
    `;
    
    return of(mockPreview).pipe(delay(500));
  }
}