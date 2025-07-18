import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, delay, BehaviorSubject } from 'rxjs';

export interface TrainingStatus {
  isTraining: boolean;
  currentEpoch: number;
  totalEpochs: number;
  currentLoss: number;
  currentAccuracy: number;
  estimatedTimeRemaining: number;
}

export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: string;
  model: string;
}

@Injectable({
  providedIn: 'root'
})
export class TrainingService {
  private apiUrl = '/api/training';
  private trainingStatusSubject = new BehaviorSubject<TrainingStatus>({
    isTraining: false,
    currentEpoch: 20,
    totalEpochs: 20,
    currentLoss: 0.28,
    currentAccuracy: 0.942,
    estimatedTimeRemaining: 0
  });

  constructor(private http: HttpClient) {}

  getTrainingStatus(): Observable<TrainingStatus> {
    return this.trainingStatusSubject.asObservable();
  }

  startTraining(config: TrainingConfig): Observable<any> {
    // Update training status
    this.trainingStatusSubject.next({
      ...this.trainingStatusSubject.value,
      isTraining: true,
      currentEpoch: 0
    });

    // Simulate training progress
    this.simulateTraining();

    return of({ success: true, message: 'Training started' }).pipe(delay(1000));
  }

  stopTraining(): Observable<any> {
    this.trainingStatusSubject.next({
      ...this.trainingStatusSubject.value,
      isTraining: false
    });

    return of({ success: true, message: 'Training stopped' }).pipe(delay(500));
  }

  pauseTraining(): Observable<any> {
    this.trainingStatusSubject.next({
      ...this.trainingStatusSubject.value,
      isTraining: false
    });

    return of({ success: true, message: 'Training paused' }).pipe(delay(500));
  }

  resumeTraining(): Observable<any> {
    this.trainingStatusSubject.next({
      ...this.trainingStatusSubject.value,
      isTraining: true
    });

    return of({ success: true, message: 'Training resumed' }).pipe(delay(500));
  }

  getTrainingHistory(): Observable<any[]> {
    const mockHistory = [
      { epoch: 20, loss: 0.28, accuracy: 0.942, lr: 0.0001, time: '14:32' },
      { epoch: 19, loss: 0.31, accuracy: 0.938, lr: 0.0001, time: '14:25' },
      { epoch: 18, loss: 0.33, accuracy: 0.935, lr: 0.0001, time: '14:18' },
      { epoch: 17, loss: 0.36, accuracy: 0.928, lr: 0.0001, time: '14:11' },
      { epoch: 16, loss: 0.38, accuracy: 0.924, lr: 0.0001, time: '14:04' }
    ];

    return of(mockHistory).pipe(delay(1000));
  }

  private simulateTraining(): void {
    // Simulate training progress over time
    let currentEpoch = 0;
    const totalEpochs = 20;

    const interval = setInterval(() => {
      if (!this.trainingStatusSubject.value.isTraining) {
        clearInterval(interval);
        return;
      }

      currentEpoch++;
      const progress = currentEpoch / totalEpochs;
      
      this.trainingStatusSubject.next({
        isTraining: currentEpoch < totalEpochs,
        currentEpoch,
        totalEpochs,
        currentLoss: 0.8 - (progress * 0.52), // Decreasing loss
        currentAccuracy: 0.78 + (progress * 0.162), // Increasing accuracy
        estimatedTimeRemaining: (totalEpochs - currentEpoch) * 60 // 1 minute per epoch
      });

      if (currentEpoch >= totalEpochs) {
        clearInterval(interval);
      }
    }, 2000); // Update every 2 seconds for demo
  }
}