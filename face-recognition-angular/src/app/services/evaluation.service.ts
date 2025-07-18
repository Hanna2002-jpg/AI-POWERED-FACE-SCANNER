import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, delay } from 'rxjs';

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  eer: number;
}

export interface ModelComparison {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  status: 'best' | 'baseline' | 'comparison';
}

@Injectable({
  providedIn: 'root'
})
export class EvaluationService {
  private apiUrl = '/api/evaluation';

  constructor(private http: HttpClient) {}

  getEvaluationMetrics(): Observable<EvaluationMetrics> {
    const mockMetrics: EvaluationMetrics = {
      accuracy: 0.942,
      precision: 0.938,
      recall: 0.945,
      f1Score: 0.941,
      auc: 0.984,
      eer: 0.058
    };

    return of(mockMetrics).pipe(delay(1000));
  }

  getModelComparison(): Observable<ModelComparison[]> {
    const mockComparison: ModelComparison[] = [
      { name: 'ArcFace (Fine-tuned)', accuracy: 0.942, precision: 0.938, recall: 0.945, status: 'best' },
      { name: 'ArcFace (Baseline)', accuracy: 0.921, precision: 0.918, recall: 0.923, status: 'baseline' },
      { name: 'InsightFace', accuracy: 0.915, precision: 0.912, recall: 0.917, status: 'comparison' },
      { name: 'FaceNet', accuracy: 0.898, precision: 0.895, recall: 0.901, status: 'comparison' }
    ];

    return of(mockComparison).pipe(delay(1000));
  }

  runEvaluation(): Observable<any> {
    return of({ success: true, message: 'Evaluation completed' }).pipe(delay(3000));
  }

  generateConfusionMatrix(): Observable<any> {
    const mockMatrix = {
      truePositives: 1387,
      falsePositives: 89,
      falseNegatives: 85,
      trueNegatives: 1439
    };

    return of(mockMatrix).pipe(delay(1500));
  }

  getThresholdAnalysis(): Observable<any[]> {
    const mockAnalysis = [
      { threshold: 0.3, accuracy: 0.891, precision: 0.923, recall: 0.856 },
      { threshold: 0.4, accuracy: 0.918, precision: 0.935, recall: 0.898 },
      { threshold: 0.5, accuracy: 0.942, precision: 0.938, recall: 0.945 },
      { threshold: 0.6, accuracy: 0.935, precision: 0.952, recall: 0.916 },
      { threshold: 0.7, accuracy: 0.912, precision: 0.968, recall: 0.858 }
    ];

    return of(mockAnalysis).pipe(delay(1000));
  }
}